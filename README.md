# Audio

All-in-one audio processing and transcription software.

Takes in audio files, transcribes them with whisper.

Cleanup of transcripts with local LLM.
- remove filler words
- create full sentences

Make available for local LLM as
- embedding database for RAG
- create knowledge Graph so LLM "knows" you

# Status

- [x] transcribe audio
- [ ] cleanup transcripts
- [ ] create embeddings
- [ ] connect to local LLM with RAG
- [ ] create knowledge graph

# Requirements

Cuda enabled GPU
Postgres DB
vLLM

# My Setup

- OS: Ubuntu 22.04 LTS
- GPU: 2x RTX 4090


