import pandas as pd
import boto3
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import StringIO
import os

class Chunks:
    def __init__(self):
        self.s3_bucket_name = os.getenv('BUCKET_NAME')
        self.s3_folder_path = os.getenv('DOCUMENTATION_FOLDER_NAME') 
        self.s3_file_key = 'chunks/pdf_docs_chunks.csv'
        self.s3 = boto3.client('s3')
        self.chunk_size = 250
        self.chunk_overlap = 30

    def chunk_markdown_file_by_headers(self, markdown_text):
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(markdown_text)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        
        splits = text_splitter.split_documents(md_header_splits)
        return splits

    def read_s3_md_files_to_chunks(self, bucket_name, folder_path):
        chunks_all_files = []
        objects = self.s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)

        if 'Contents' in objects:
            for obj in objects['Contents']:
                file_key = obj['Key']
                if file_key.endswith('.md'):
                    file_obj = self.s3.get_object(Bucket=bucket_name, Key=file_key)
                    md_content = file_obj['Body'].read().decode('utf-8')
                    chunks = self.chunk_markdown_file_by_headers(md_content)
                    chunks_all_files.extend(chunks)

        return chunks_all_files
    
    def consolidate_all_chunks_to_csv_and_upload_to_s3(self):
        # Read chunks from the specified folder in the S3 bucket
        print("Reading MD files from S3 and splitting into chunks...")
        all_chunks = self.read_s3_md_files_to_chunks(self.s3_bucket_name, self.s3_folder_path)
        data = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in all_chunks]
        pdf_docs_chunks = pd.DataFrame(data)
        print("Writing chunks to CSV...")
        pdf_docs_chunks.to_csv('pdf_docs_chunks.csv', index=False)
        print("Uploading CSV to S3...")
        self.s3.upload_file('pdf_docs_chunks.csv', self.s3_bucket_name, self.s3_file_key)
    
    def download_all_csv_chunks_from_s3(self):
        print("Downloading CSV chunks from S3...")
        file_obj = self.s3.get_object(Bucket=self.s3_bucket_name, Key=self.s3_file_key)
        file_content = file_obj['Body'].read().decode('utf-8')
        csv_file = StringIO(file_content)
        pdf_docs_chunks = pd.read_csv(csv_file)
        pdf_docs_chunks.head()
        self.s3.download_file(self.s3_bucket_name, self.s3_file_key, 'pdf_docs_chunks.csv')
        return pd.read_csv('pdf_docs_chunks.csv')