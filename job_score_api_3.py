from quart import Quart, request, jsonify
from pydantic import BaseModel
from pydantic import BaseModel, RootModel
from openai import AsyncOpenAI
import json
from typing import Dict, Any, Optional, List, Union
import os
from dotenv import load_dotenv
load_dotenv()



app = Quart(__name__)


class JobMatch(BaseModel):
    job_id: str
    score: int

class JobMatchResult(RootModel):
    root: Dict[str, int]

# Initialize AsyncOpenAI client
api = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api)

class JobMatcherError(Exception):
    """Custom exception for job matching errors"""
    pass

async def parse_json_input(data: Union[str, dict, list]) -> Any:
    """Parse JSON input and handle potential errors"""
    if isinstance(data, (dict, list)):
        return data
    
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        raise JobMatcherError(f"Invalid JSON format: {str(e)}")


def create_matching_prompt(results: List[Dict[str, Any]], user_skill: Dict[str, Any]) -> str:
    """Create the prompt for GPT-4 matching"""
    return f"""

    You are an AI assistant specialized in matching job postings to member profiles. Your objective is to evaluate each job posting against the provided member profile and assign a raw match score based on the comprehensive criteria outlined below. Ensure that the final scores accurately reflect both lenient and strict matching where appropriate. If a job posting does not meet any of the matching criteria, assign a low score based on the match quality instead of assigning a score of 0.
    
    **Input Data**
    
    a) **Job Postings**
    
    The `job_post_skills` variable is a list of dictionaries structured as follows:
    {json.dumps(results, indent=2)}
    
    b) **Member Profile**
    
    The `user_skill` variable is a dictionary representing a member's skills in the following structure:
    {json.dumps(user_skill, indent=2)}
    
    **Matching Criteria and Weights:**
    
    **Skills (30% Weight)**
    
    - **Required Skills:**
      - Exact Match: +30 points
      - Related Match: +20 points
      - Partial Match: +15 points
    - **Preferred Skills:**
      - Exact Match: +10 points
      - Related Match: +5 points
    
    **Other Skills (25% Weight)**
    
    - **Tools, Certifications, Technologies:**
      - Exact Match: +25 points
      - Related Match: +15 points
    
    **Qualifications (15% Weight)**
    
    - **Education and Certifications:**
      - Exact Match: +15 points
      - Higher Qualification: +10 points
      - Related or Lower Qualification: +5 points
      - Partial Match: +7 points
    
    **Key Responsibilities (15% Weight)**
    
    - **Matched Responsibilities:**
      - Exact Match: +15 points
      - Partial Match: +7 points
    
    **Industry (5% Weight)**
    
    - **Industry Match:**
      - Exact Match: +5 points
      - Related Experience: +3 points
    
    **Role (5% Weight)**
    
    - **Role Match:**
      - Exact Match: +5 points
      - Related or Similar Role: +3 points
    
    **Location (5% Weight)**
    
    - **Location Match:**
      - Exact City Match: +5 points
      - Nearby Location: +3 points
    
    **Normalization Formula:**
    Scaled Score = (Candidate's Raw Score / Highest Raw Score) * 100
    
    **Task:**
    
    1. **Determine Job Experience Level:**
       - Analyze the Job Postings to identify the required Experience Level (e.g., "Entry Level", "Mid Level", "Senior Level").
       - Based on the Experience Level, adjust the matching strictness:
         - **Entry-Level Positions:** Apply more lenient matching criteria.
         - **Experienced Positions (Mid/Senior Level):** Apply stricter matching criteria, ensuring that job postings require relevant and substantial experience.
    
    2. **Detailed Evaluation:**
       - For each job posting, evaluate it against each matching criterion.
       - **For Experienced Positions Only:**
         - Ensure that the job posting requires a similar or higher level of experience compared to the member's profile.
         - If a job posting demands expertise or roles that exceed the member's qualifications, adjust the score accordingly.
       - Assign an integer raw score for each criterion based on how well the job posting meets the member's qualifications and experience.
    
    3. **Weighted Scoring:**
       - Ensure the sum of all criterion scores reflects the allocated weights, totaling up to the maximum possible points (e.g., 100 points) for each job posting.
       - All individual criterion scores must be integers and should not exceed their maximum allocated points.
    
    4. **Identify Highest Raw Score:**
       - Determine the highest raw score among all job postings for the member.
    
    5. **Score Normalization:**
       - Normalize each job posting's raw score to a 0-100 range using the provided formula.
       - Round the scaled scores to the nearest integer.
       - **No minimum threshold:** Scores can range below 85 if the match quality warrants it.
    
    6. **Filtering and Ranking:**
       - Select all job postings with **Scaled Scores** of **85 or higher**.
       - Rank these job postings in descending order based on their **Scaled Scores**.
       - In case of tie scores, prioritize job postings with higher scores in the more critical categories:
         - Skills (30%)
         - Other Skills (25%)
         - Qualifications (15%)
       - Ensure that all job postings are included in the output, even if their scores are below 85.
    
    7. **Output:**
       - Provide the final list of scored job postings in the specified JSON format.
       - Ensure that all job postings in the input are included in the output, even if the score is below 85.
       - Only output the JSON object. Do not include any additional text or explanations.
        

        
        - **Example Output:**
        {{
            "33": 88,
            "35": 90,
            "37": 85,
            "38": 87,
            "39": 89,
            "40": 92,
            "10": 70,
            "11": 65,
            "12": 60,
            "15": 55,
            "16": 50,
        }}
    
    **Important Instructions:**
        
        - **Output Only JSON:** Do not include any reasoning, explanations, or additional text. The response must be only the JSON object as shown in the example.
        
        - **Consistent Formatting:** Ensure that the JSON output strictly follows the provided structure with accurate indentation and no syntax errors.
        
        - **Deterministic Output:** Maintain temperature=0.0 to ensure the model produces the same output for the same input every time.
        
        - **Validation:** After generating the JSON, perform a quick validation to ensure:
          - All job posting IDs from the input are present in the output.
          - Scaled scores are between 0 and 100.
          - Job postings that do not meet any matching criteria have appropriately low scores based on the match quality.
          - The JSON structure is correct and free of syntax errors.
        
        **Experience-Based Matching Adjustment:**
        
        - **Entry-Level Positions:** Apply more lenient matching criteria to accommodate job postings that are open to candidates who may be early in their careers but show potential.
        
        - **Experienced Positions:**
          - Apply stricter matching criteria to ensure job postings require the necessary expertise and track record.
          - **Mandatory Requirement:** Job postings must align with the member's experience level. If a job posting requires experience beyond the member's qualifications, adjust the score accordingly.
        
        Based on the above criteria and using detailed, multi-step reasoning with score normalization, evaluate and assign raw match scores to each job posting for the given member profile. Ensure that all job posting IDs are included in the final JSON output with their respective scores, which can be below or above 85 based on the match quality.
    
    """



async def get_gpt4_response(prompt: str) -> JobMatchResult:
    """Get response from GPT-4 asynchronously and parse it"""
    try:
        response = await client.chat.completions.create(
            model="chatgpt-4o-latest",  # Update this to the latest model version you have access to
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that matches job postings to member profiles."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            response_format={"type": "json_object"},
            max_tokens=8000
        )
        
        # Parse the response using the JobMatchResult model
        return JobMatchResult.model_validate_json(response.choices[0].message.content)
    except Exception as e:
        raise JobMatcherError(f"GPT-4 API error: {str(e)}")



async def process_job_matching(results: List[Dict[str, Any]], member_profile: Dict[str, Any]) -> Dict[str, int]:
    """Process job matching with GPT-4 and additional post-processing"""
    prompt = create_matching_prompt(results, member_profile)
    # print(prompt)
    gpt4_result = await get_gpt4_response(prompt)
    
    # Post-processing to ensure consistency
    max_score = max(gpt4_result.root.values())
    normalized_matches = {}
    
    for job_id, score in gpt4_result.root.items():
        # Normalize to 0-100 scale
        normalized_score = int((score / max_score) * 100)
        # Round to nearest 5
        rounded_score = round(normalized_score / 5) * 5
        normalized_matches[job_id] = rounded_score
    
    return normalized_matches    

@app.route('/match-jobs', methods=['POST'])
async def match_jobs():
    """Main endpoint for job matching"""
    try:
        # ... (previous code remains the same)
        # Get JSON data from request
        data = await request.get_json()
        if not data or 'job_post_json' not in data or 'user_profile_json' not in data:
            return jsonify({"error": "Missing required JSON data"}), 400

        # Parse input JSON - now handles both string and direct JSON
        job_posts = await parse_json_input(data['job_post_json'])
        user_profile = await parse_json_input(data['user_profile_json'])

        # Extract user skills
        user_skills = user_profile
        
        # Process job posts and get results
        results = job_posts
        
        # Get matching results using the processed results
        result = await process_job_matching(results, user_skills)
        
        return jsonify(result)

    except JobMatcherError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.errorhandler(404)
async def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
async def server_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5002, debug=True, use_reloader=False)
