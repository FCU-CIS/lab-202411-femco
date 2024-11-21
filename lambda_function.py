import json
import os
import boto3

AGENT_ID = os.environ["AGENT_ID"]
AGENT_ALIAS_ID = os.environ["AGENT_ALIAS_ID"]

client = boto3.client("bedrock-agent-runtime")

def lambda_handler(event, context):
    try:
        query = event["queryStringParameters"]
        text = query["text"]
        request_id = event["requestContext"]["requestId"]

        agent_response, chunks = ask_agent(request_id, text)

        return {
            "statusCode": 200,
            "body": json.dumps(
                {"response": agent_response, "chunks": chunks},
            ),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }
    
def ask_agent(session_id: str, text: str):
    response = client.invoke_agent(
        agentId=AGENT_ID,
        agentAliasId=AGENT_ALIAS_ID,
        sessionId=session_id,
        inputText=text,
    )

    chunks = []
    response_text = ""

    for event in response["completion"]:
        chunk = event.get("chunk")
        if chunk:
            message = chunk.pop("bytes").decode()
            response_text += message

            chunks.append(chunk)

    response_text += "\n"

    return response_text, chunks