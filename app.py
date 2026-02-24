import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from textblob import TextBlob
from typing import List
import googleapiclient.discovery
import googleapiclient.errors

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────

# FIX: Changed from "YT_API" to "YOUTUBE_API_KEY" to match your export
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Initialize YouTube client
youtube = None
if YOUTUBE_API_KEY:
    try:
        youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        print("✓ YouTube API initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing YouTube API: {e}")
else:
    print("✗ YOUTUBE_API_KEY environment variable not found")

app = FastAPI(title="YouTube Sentiment API")

# FIX: Added localhost to allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  Models & Helpers
# ─────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    url: str
    limit: int = 100  # Default to 100, max 1000

class CommentResponse(BaseModel):
    comment_id: str
    text: str
    sentiment: str
    polarity: float

def extract_video_id(url: str):
    """Extracts the video ID from various YouTube URL formats."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:shorts\/)([0-9A-Za-z_-]{11})',
        r'^([0-9A-Za-z_-]{11})$'  # Just the ID
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_sentiment(text: str):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.05:
        category = "Positive"
    elif polarity < -0.05:
        category = "Negative"
    else:
        category = "Neutral"
    return category, round(polarity, 4)

# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────

@app.get("/")
def root():
    """Serve the frontend HTML."""
    return FileResponse("index.html")

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "Running",
        "youtube_api_configured": youtube is not None,
        "api_key_present": YOUTUBE_API_KEY is not None
    }

@app.post("/analyze_comments/", response_model=List[CommentResponse])
def analyze_comments(request: AnalysisRequest):
    """
    Fetch and analyze comments from a YouTube video.
    
    Args:
        request: Contains YouTube URL and limit for number of comments
        
    Returns:
        List of comments with sentiment analysis
    """
    # Check if YouTube API is configured
    if not youtube:
        raise HTTPException(
            status_code=500,
            detail="YouTube API Key not configured. Please set YOUTUBE_API_KEY environment variable."
        )
    

    video_id = extract_video_id(request.url)
    if not video_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube URL. Please provide a valid YouTube video link."
        )

    comments = []
    next_page_token = None
    total_to_fetch = min(request.limit, 1000)

    try:
        while len(comments) < total_to_fetch:

            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, total_to_fetch - len(comments)),
                pageToken=next_page_token,
                textFormat="plainText",
                order="relevance"  
            ).execute()

        
            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                text = snippet["textDisplay"]
                cid = item["id"]
                
                
                sentiment, polarity = get_sentiment(text)
                
                comments.append(CommentResponse(
                    comment_id=cid,
                    text=text[:500],  # Truncate long comments
                    sentiment=sentiment,
                    polarity=polarity
                ))

            # Check for more pages
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        if not comments:
            raise HTTPException(
                status_code=404,
                detail="No comments found for this video. The video may have comments disabled."
            )

        return comments

    except googleapiclient.errors.HttpError as e:
        error_details = str(e)
        
        # Handle specific error cases
        if e.resp.status == 403:
            if "commentsDisabled" in error_details:
                raise HTTPException(
                    status_code=403,
                    detail="Comments are disabled for this video."
                )
            elif "quotaExceeded" in error_details:
                raise HTTPException(
                    status_code=429,
                    detail="YouTube API quota exceeded. Please try again later."
                )
            else:
                raise HTTPException(
                    status_code=403,
                    detail="Access forbidden. Please check your API key permissions."
                )
        elif e.resp.status == 404:
            raise HTTPException(
                status_code=404,
                detail="Video not found. Please check the URL."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"YouTube API Error: {error_details}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)