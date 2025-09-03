from react_agent.doc_processing.doc_extraction import DocumentProcessor, ProcessedDocument, TextCleaner
from react_agent.doc_processing.doc_chunking import BaseChunker, TokenBasedChunker, SemanticChunker, ChunkingManager, ChunkingStrategy
from react_agent.doc_processing.doc_embedding import EmbeddingGenerator
from react_agent.doc_processing.dbase_store import VectorDatabase, DatabaseConfig, VectorRecord
import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import mimetypes
import logging
import json
import asyncio

#Intialize VectorDB
db_config = DatabaseConfig
vector_db = VectorDatabase(db_config)

async def main():
    # Set up logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    # Initialize processor
    processor = DocumentProcessor()
    cleaner = TextCleaner()
    chunker = ChunkingManager()
    embedding_generator = EmbeddingGenerator()

    # Example: Process a single file
    try:
        # Replace with your actual file path
        file_path = Path(r"C:\Users\dv146ms\Downloads\Invoice_000001.pdf").resolve()

        
        if Path(file_path).exists():
            processed_doc = processor.process_document(file_path)
            if processed_doc.content:
                source_file = processed_doc.source_file
                content = processed_doc.content
                doc_metadata = processed_doc.metadata
                print(f"Successfully processed: {source_file}")
                print(f"Content length: {len(content)} characters")
                print(f"Processing time: {processed_doc.processing_time:.2f} seconds")
                print(f"Metadata: {doc_metadata}")
            
                # Apply advanced cleaning
                enhanced_content = cleaner.remove_headers_footers(processed_doc.content)
                enhanced_content = cleaner.normalize_spacing(enhanced_content)
                enhanced_content = cleaner.fix_hyphenation(enhanced_content)

                #Chunk document content
                print('---Chunking document---...')
                doc_chunks = chunker.chunk_text(enhanced_content, ChunkingStrategy.SEMANTIC, source_file)
                doc_chunks = doc_chunks[0].content
                #Get Document embeddings
                print('---Embedding document---...')
                doc_embeddings = embedding_generator.generate_embedding(doc_chunks)
                
                #Saving to vector database
                print('---Saving document to vector database---...')
                try:
                    await vector_db.initialize()
                    
                    # Health check
                    health = await vector_db.health_check()
                    print(f"Database health: {health}")
                    
                    # Create sample records
                    records = [
                        VectorRecord(
                            content=doc_chunks,
                            embedding=doc_embeddings,
                            metadata=doc_metadata,
                            source=source_file,
                            chunk_index=0
                        )
                    ]
                    
                    # Insert records
                    stats = await vector_db.insert_vectors(records)
                    print(f"Insertion stats: {stats}")
                    
                    # Perform similarity search
                    query_embedding = doc_embeddings
                    results = await vector_db.similarity_search(json.dumps(query_embedding), limit=5, threshold=0.5)
                    print(f"Search results: {len(results)} documents found")
                    
                    # Get database statistics
                    db_stats = await vector_db.get_stats()
                    print(f"Database stats: {db_stats}")
                    
                    # Optimize database
                    await vector_db.optimize_database()
                    
                except Exception as e:
                        logger.error(f"Error in main: {e}")    
                finally:
                        await vector_db.close()

            else:
                print(f"Failed to process: {processed_doc.errors}")
        
        # Example: Process a directory
        # processed_docs = processor.process_directory("./documents", recursive=True)
        # stats = processor.get_processing_stats(processed_docs)
        # print(f"Processing statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Processing error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
