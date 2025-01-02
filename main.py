import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import os
import psycopg2
from datetime import datetime
from pathlib import Path
from pydub import AudioSegment
import tempfile
import re
from typing import List

def convert_to_flac(input_path: str) -> str:
    """Converts audio to FLAC and returns path to temporary file"""
    audio = AudioSegment.from_file(input_path)
    temp_path = tempfile.mktemp(suffix='.flac')
    audio.export(temp_path, format='flac')
    return temp_path

def transcribe_audio(file_path: str):
    flac_path = convert_to_flac(file_path)
    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            torch_dtype=torch.float16,
            device="cuda:0",
            model_kwargs={"attn_implementation": "flash_attention_2"} 
            if is_flash_attn_2_available() 
            else {"attn_implementation": "sdpa"},
        )
        
        # Move model to GPU after initialization
        pipe.model = pipe.model.to("cuda:0")
        
        # Language parameter moved to the generation call
        return pipe(
            flac_path,
            chunk_length_s=30,
            batch_size=24,
            return_timestamps=True,
            generate_kwargs={"language": "en"}  # Language parameter goes here
        )
    finally:
        os.remove(flac_path)

def first_step():
    db = AudioDatabase("rec")
    unprocessed = db.get_unprocessed_files()
    print(f"Found {len(unprocessed)} files to process")
    
    for file_path in unprocessed:
        try:
            result = transcribe_audio(file_path)
            db.store_transcription(file_path, result)
            print(f"Successfully processed: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")


class AudioDatabase:
    def __init__(self, audio_folder):
        self.audio_folder = audio_folder
        self.conn = psycopg2.connect(
            dbname="audio_notes",
            user="postgres",
            password="postgres",
            host="localhost"
        )
        self.cursor = self.conn.cursor()
        self._ensure_tables_exist()
    
    def _ensure_tables_exist(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id SERIAL PRIMARY KEY,
                filepath VARCHAR(255) UNIQUE NOT NULL,
                filesize BIGINT NOT NULL,
                date_created TIMESTAMP NOT NULL,
                duration_seconds FLOAT NOT NULL,
                processed BOOLEAN DEFAULT FALSE
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                id SERIAL PRIMARY KEY,
                file_id INTEGER REFERENCES files(id),
                date_recorded DATE NOT NULL,
                time_from TIME NOT NULL,
                time_to TIME NOT NULL,
                transcript TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
    
    def get_file_metadata(self, filepath):
        audio = AudioSegment.from_file(filepath)
        return {
            'filepath': str(filepath),
            'filesize': os.path.getsize(filepath),
            'date_created': datetime.fromtimestamp(os.path.getctime(filepath)),
            'duration_seconds': len(audio) / 1000.0
        }
    
    def get_unprocessed_files(self):
        audio_files = []
        for file in Path(self.audio_folder).glob('*.m4a'):
            metadata = self.get_file_metadata(file)
            
            self.cursor.execute("""
                INSERT INTO files (filepath, filesize, date_created, duration_seconds)
                VALUES (%(filepath)s, %(filesize)s, %(date_created)s, %(duration_seconds)s)
                ON CONFLICT (filepath) DO NOTHING
            """, metadata)
        
        self.cursor.execute("""
            SELECT filepath FROM files 
            WHERE processed = FALSE
        """)
        self.conn.commit()
        
        return [row[0] for row in self.cursor.fetchall()]
    
    def __del__(self):
        self.cursor.close()
        self.conn.close()

    def store_transcription(self, filepath: str, transcription_result: dict):
        # Get file_id
        self.cursor.execute("SELECT id FROM files WHERE filepath = %s", (filepath,))
        file_id = self.cursor.fetchone()[0]
        
        # Date parsing code remains the same...
        date_str = os.path.basename(filepath).split('.')[0]
        try:
            date_recorded = datetime.strptime(date_str, '%Y_%m_%d').date()
        except ValueError:
            try:
                date_recorded = datetime.strptime(date_str, '%d_%m_%Y').date()
            except ValueError:
                raise ValueError(f"File name {date_str} must be in YYYY_MM_DD or DD_MM_YYYY format")
        
        # Store each chunk with its timestamps, with proper error handling
        for chunk in transcription_result['chunks']:
            # Skip chunks with invalid timestamps
            if not chunk['timestamp'] or None in chunk['timestamp']:
                continue
            
            try:
                # Simple conversion without timezone handling
                time_from = datetime.fromtimestamp(chunk['timestamp'][0]).time()
                time_to = datetime.fromtimestamp(chunk['timestamp'][1]).time()
                
                self.cursor.execute("""
                    INSERT INTO transcripts 
                    (file_id, date_recorded, time_from, time_to, transcript, model)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    file_id,
                    date_recorded,
                    time_from,
                    time_to,
                    chunk['text'],
                    'whisper-large-v3'
                ))
            except (TypeError, ValueError) as e:
                print(f"Warning: Skipping chunk due to invalid timestamp: {e}")
                continue
        
        # Mark file as processed
        self.cursor.execute("""
            UPDATE files SET processed = TRUE 
            WHERE id = %s
        """, (file_id,))
        
        self.conn.commit()

class PostProcessing:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname="audio_notes",
            user="postgres",
            password="postgres",
            host="localhost"
        )
        self.cursor = self.conn.cursor()
        self._ensure_tables_exist()
    
    def _ensure_tables_exist(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentences (
                id SERIAL PRIMARY KEY,
                file_id INTEGER REFERENCES files(id),
                sentence TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
    
    def process_file_transcripts(self, file_id: int) -> List[str]:
        # Get all transcripts for the file, ordered by time
        self.cursor.execute("""
            SELECT transcript 
            FROM transcripts 
            WHERE file_id = %s 
            ORDER BY time_from
        """, (file_id,))
        
        # Combine all transcripts into one text
        transcripts = self.cursor.fetchall()
        full_text = ' '.join(transcript[0] for transcript in transcripts)
        
        # Split into sentences and clean them
        sentences = [s.strip() for s in re.split(r'[.!?]+', full_text) if s.strip()]
        
        # Store sentences
        for sentence in sentences:
            word_count = len(sentence.split())
            self.cursor.execute("""
                INSERT INTO sentences (file_id, sentence, word_count)
                VALUES (%s, %s, %s)
            """, (file_id, sentence, word_count))
        
        self.conn.commit()
        return sentences
    
    def process_all_unprocessed(self):
        # Get all files that have transcripts but no sentences
        self.cursor.execute("""
            SELECT DISTINCT f.id 
            FROM files f
            JOIN transcripts t ON f.id = t.file_id
            LEFT JOIN sentences s ON f.id = s.file_id
            WHERE s.id IS NULL AND f.processed = TRUE
        """)
        
        file_ids = [row[0] for row in self.cursor.fetchall()]
        
        for file_id in file_ids:
            try:
                sentences = self.process_file_transcripts(file_id)
                print(f"Processed file {file_id}: {len(sentences)} sentences extracted")
            except Exception as e:
                print(f"Error processing file {file_id}: {e}")
    
    def __del__(self):
        self.cursor.close()
        self.conn.close()




if __name__ == "__main__":
    # First process audio files
    #first_step()
    # Then process transcripts into sentences
    print("\nProcessing transcripts into sentences...")
    post = PostProcessing()
    post.process_all_unprocessed()
