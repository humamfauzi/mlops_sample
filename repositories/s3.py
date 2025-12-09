import boto3
import json
import pickle
import io
from typing import List

from .struct import TransformationInstruction, TransformationObject, ModelObject
import os
from dotenv import load_dotenv

class S3:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        load_dotenv()

        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_session_token = os.getenv("AWS_SESSION_TOKEN")
        region_name = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")

        client_kwargs = {}
        if aws_access_key_id and aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = aws_access_key_id
            client_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            client_kwargs["aws_session_token"] = aws_session_token
        if region_name:
            client_kwargs["region_name"] = region_name

        self.s3_client = boto3.client("s3", **client_kwargs)

    def save_transformation_instruction(self, run_id: str, instructions: List[TransformationInstruction]):
        all_instruction = [inst.to_dict() for inst in instructions]
        key = f"{run_id}/transformation/instruction.json"

        json_data = json.dumps(all_instruction, indent=2, ensure_ascii=False)
        
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=json_data.encode('utf-8'),
            ContentType='application/json'
        )
        return f"s3://{self.bucket_name}/{key}"

    def save_transformation_object(self, run_id: str, transformation_objects: List[TransformationObject]):
        s3_paths = []
        for obj in transformation_objects:
            key = f"{run_id}/transformation/objects/{obj.filename}"

            buffer = io.BytesIO()
            pickle.dump(obj.object, buffer, protocol=pickle.HIGHEST_PROTOCOL)
            buffer.seek(0)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=buffer.getvalue(),
                ContentType='application/octet-stream'
            )
            s3_paths.append(f"s3://{self.bucket_name}/{key}")
        return s3_paths

    def load_transformation_instruction(self, run_id: str):
        key = f"{run_id}/transformation/instruction.json"
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
        data = json.loads(response['Body'].read().decode('utf-8'))
        
        return [TransformationInstruction(**item) for item in data]

    def load_transformation_object(self, run_id: str, transformation_id: str = "", ttype: str = ""):
        prefix = f"{run_id}/transformation/objects/{transformation_id}.{ttype}"
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=prefix)
        data = pickle.loads(response['Body'].read())
        return TransformationObject(filename=transformation_id, object=data)

    def save_model(self, run_id:str,  model: ModelObject):
        key = f"{run_id}/model/{model.filename}"
        
        # Serialize model using pickle
        buffer = io.BytesIO()
        pickle.dump(model.object, buffer, protocol=pickle.HIGHEST_PROTOCOL)
        buffer.seek(0)
        
        # Upload to S3
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=buffer.getvalue(),
            ContentType='application/octet-stream'
        )
        return f"s3://{self.bucket_name}/{key}"

    def load_model(self, parent_run_id:int, name: str):
        key = f"{parent_run_id}/model/{name}"
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
        model = pickle.loads(response['Body'].read())
        return ModelObject(filename=name, object=model)


    
