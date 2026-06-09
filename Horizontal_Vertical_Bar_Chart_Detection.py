# 1. Import the library
from inference_sdk import InferenceHTTPClient

# 2. Connect to your workflow
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="G0xjBBPTGJe4HsRjyQpW"
)

# 3. Run your workflow on an image
result = client.run_workflow(
    workspace_name="khas-workspace-3cwa2",
    workflow_id="bar-chart-detection-and-crop-1779799226718",
    images={
        "image": "/Users/nguyenanhvu/Documents/AMD-Semester3/GroupProject/ibcs-ml-base/Dataset/Compliant/8.png" # Path to your image file
    },
    use_cache=True # Speeds up repeated requests
)

# 4. Get your results
print(result)
