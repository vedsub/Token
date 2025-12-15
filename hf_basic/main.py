from transformers import pipeline


pipe = pipeline("image-text-to-text",model="google/gemma-3-4b-it")
messages = [
    
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]
pipe(text = messages)
