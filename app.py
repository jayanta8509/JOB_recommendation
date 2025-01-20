from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
import pyodbc
from datetime import datetime
from decimal import Decimal
import pyodbc
from datetime import datetime
from typing import List, Dict, Any
from flask import Flask, jsonify
from decimal import Decimal
from openai import AsyncOpenAI 
import logging
import os
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
# Load BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # When you used frist time it tske some time to download the model and tokenizer
model = AutoModel.from_pretrained('bert-base-uncased')
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use mean pooling
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    return mean_embeddings.numpy()

def create_job_text(job):
    fields = [
        job.get("JobTitle", ""),
        job.get("EmploymentTypeName", ""),
        job.get("CityName", ""),
        job.get("Job_Description", ""),
        job.get("JobLocation", "")
    ]
    text = " ".join([str(field) for field in fields if field])
    return text.strip() if text.strip() else None

def create_member_text(member):
    fields = [
        member.get("MemberTypeName", ""),
        member.get("Id", ""),
        member.get("GenderName", ""),
        member.get("EthnicityName", ""),
        member.get("PronounName", ""),
        member.get("AgeName", ""),
        member.get("SexualOrientationName", ""),
        member.get("NationalityName", ""),
        member.get("JobLevelName", ""),
        member.get("EmploymentTypeName", ""),
        member.get("IndustryName", ""),
        member.get("StateName", ""),
        member.get("CityName", ""),
        member.get("CommunicationName", ""),
        member.get("LeadershipName", ""),
        member.get("CriticalThinkingNames", ""),
        member.get("GrowthMindsetNames", ""),
        member.get("CollaborationNames", ""),
        member.get("MetacognitionNames", ""),
        member.get("CharacterNames", ""),
        member.get("CreativityNames", ""),
        member.get("FortitudeNames", ""),
        member.get("TechnicalSkillNames", ""),
        member.get("MindfulnessNames", ""),
        member.get("RaceNames", ""),
        member.get("PersonalFamilyNames", ""),
        member.get("ExperienceDetails", ""),
        member.get("MemberEducationDetails", "")
    ]
    text = " ".join([str(field) for field in fields if field])
    return text.strip() if text.strip() else None

def get_job_post_details(job):
    """Get formatted job post details as a dictionary."""
    details = {
        "Job Id": job.get("Id", "Unknown"),
        "Job Code": job.get("JobCode", "Unknown"),
        "Job Title": job.get("JobTitle", "Unknown"),
        "Employment Type": job.get("EmploymentTypeName", "Unknown"),
        "City": job.get("CityName", "Unknown"),
        "Posted Date": job.get("PostedDate", "Unknown"),
        "Deadline for Applications": job.get("Deadline_for_Applications", "Unknown"),
        "Salary Range Start": job.get("Salary_Range_Start", "Unknown"),
        "Salary Range End": job.get("Salary_Range_End", "Unknown"),
        "Job Location": job.get("JobLocation", "Unknown"),
        "Is Active": job.get("IsActive", "Unknown"),
        "Created Date": job.get("CreatedDate", "Unknown"),
        "Modified Date": job.get("ModifiedDate", "Unknown")
    }
    return details


def get_member_details(member):
    """Get formatted member details as a dictionary."""
    details = {
        "Member Id": member.get("MemberId", "Unknown"),
        "Member First Name": member.get("MemberFirstName", "Unknown"),
        "Member Last Name": member.get("MemberLastName", "Unknown"),
        "Member Email": member.get("MemberEmail", "Unknown"),
        "Member Type": member.get("MemberTypeName", "Unknown"),
        "Gender": member.get("GenderName", "Unknown"),
        "Ethnicity": member.get("EthnicityName", "Unknown"),
        "Pronouns": member.get("PronounName", "Unknown"),
        "Age Group": member.get("AgeName", "Unknown"),
        "Sexual Orientation": member.get("SexualOrientationName", "Unknown"),
        "Nationality": member.get("NationalityName", "Unknown"),
        "Job Level": member.get("JobLevelName", "Unknown"),
        "Employment Type": member.get("EmploymentTypeName", "Unknown"),
        "Industry": member.get("IndustryName", "Unknown"),
        "State": member.get("StateName", "Unknown"),
        "City": member.get("CityName", "Unknown"),
        "Communication": member.get("CommunicationName", "Unknown"),
        "Leadership": member.get("LeadershipName", "Unknown"),
        "Critical Thinking": member.get("CriticalThinkingNames", "Unknown"),
        "Growth Mindset": member.get("GrowthMindsetNames", "Unknown"),
        "Collaboration": member.get("CollaborationNames", "Unknown"),
        "Metacognition": member.get("MetacognitionNames", "Unknown"),
        "Character": member.get("CharacterNames", "Unknown"),
        "Creativity": member.get("CreativityNames", "Unknown"),
        "Fortitude": member.get("FortitudeNames", "Unknown"),
        "Technical Skills": member.get("TechnicalSkillNames", "Unknown"),
        "Rank": member.get("Rank", "N/A"),
        "Similarity": member.get("Similarity", "N/A")
    }
    return details

def calculate_similarity(emb1, emb2):
    return cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]


async def explain_similarity(job_details, member_details, api_key):
    client = AsyncOpenAI(api_key=api_key)

    prompt = f"""
    Job Details:
    Title: {job_details['Job Title']}
    Employment Type: {job_details['Employment Type']}
    Location: {job_details['City']}
    Description: {job_details.get('Job_Description', 'Not provided')}

    Candidate Details:
    Age Group: {member_details['Age Group']}
    Education: {member_details.get('MemberEducationDetails', 'Not provided')}
    Experience: {member_details.get('ExperienceDetails', 'Not provided')}
    Skills:
    - Character: {member_details['Character']}
    - Collaboration: {member_details['Collaboration']}
    - Communication: {member_details['Communication']}
    - Creativity: {member_details['Creativity']}
    - Critical Thinking: {member_details['Critical Thinking']}
    - Growth Mindset: {member_details['Growth Mindset']}
    - Leadership: {member_details['Leadership']}
    - Metacognition: {member_details['Metacognition']}
    - Technical Skills: {member_details['Technical Skills']}
    
    The similarity score between this job and candidate is {member_details['Similarity']}%.

    Task: Analyze the job requirements and the candidate's profile to explain this similarity score. 
    Consider the following in your explanation:
    1. How well do the candidate's skills and experiences align with the job requirements?
    2. Are there any standout qualities that make this candidate particularly suitable (or unsuitable) for the role?
    3. What potential strengths or weaknesses does this match suggest?
    4. How might the candidate's age group and education level factor into their suitability for this position?
    5. Are there any skills or experiences the candidate might need to develop further for this role?

    Provide a concise yet comprehensive explanation in 3-4 sentences.
    """

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are an AI assistant specializing in HR and recruitment. Your task is to analyze job-candidate matches and provide insightful explanations for similarity scores."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        
        gpt_response = response.choices[0].message.content
    except Exception as e:
        gpt_response = f"Error in generating GPT response: {str(e)}"

    member_details["gpt_response"] = gpt_response
    return member_details

async def process_recommendations(job_id: int, openai_api_key: str) -> Any:
    try:
        # Your existing database fetching code
        def fetch_data_sync():
            try:
                conn = pyodbc.connect(
                    'DRIVER={SQL Server};'
                    'SERVER=DESKTOP-68FBO4B;'
                    'DATABASE=MPOWER_TEST;'
                    'Trusted_Connection=yes;'  # This is for local database connection using Windows Authentication, you can change accordingly for your database
                )
                cursor = conn.cursor()

                queries = {
                    "jobpost": """
                        SELECT 
                            jp.Id,
                            jp.JobCode, 
                            jp.JobTitle, 
                            jp.EmploymentTypeId, 
                            ISNULL(et.EmploymentTypeName, 'Unknown') AS EmploymentTypeName, 
                            jp.CityId, 
                            ISNULL(c.CityName, 'Unknown') AS CityName, 
                            jp.PostedDate, 
                            jp.Deadline_for_Applications,
                            jp.Salary_Range_Start, 
                            jp.Salary_Range_End, 
                            jp.JobLocation, 
                            jp.IsActive, 
                            jp.CreatedDate, 
                            jp.ModifiedDate
                        FROM JobPost AS jp
                        LEFT JOIN EmploymentType AS et ON jp.EmploymentTypeId = et.Id
                        LEFT JOIN City AS c ON jp.CityId = c.Id
                        WHERE jp.id= ?
                    """,
                    "member": """
                        SELECT
                            m.Id AS MemberId,
                            m.MemberFirstName,
                            m.MemberLastName,
                            m.MemberEmail,
                            m.IsActive,
                            m.CreatedDate,
                            m.ModifiedDate,
                            m.MemberEthnicityId,
                            m.MemberGenderId,
                            m.Age_Id,
                            a.AgeName,
                            m.CityId,
                            c.CityName,
                            m.Communication_Id,
                            Eth.EthnicityName,
                            ISNULL(comm.CommunicationName, 'Unknown') AS CommunicationName,
                            m.Employment_Type_Id,
                            ISNULL(et.EmploymentTypeName, 'Unknown') AS EmploymentTypeName,
                            m.Industry_Id,
                            ISNULL(i.IndustryName, 'Unknown') AS IndustryName,
                            m.Job_Level_Id,
                            ISNULL(jl.JobLevelName, 'Unknown') AS JobLevelName,
                            m.Leadership_Id,
                            ISNULL(l.LeadershipName, 'Unknown') AS LeadershipName,
                            m.MemberTypeId,
                            ISNULL(mt.MemberTypeName, 'Unknown') AS MemberTypeName,
                            m.Nationality_Id,
                            ISNULL(n.NationalityName, 'Unknown') AS NationalityName,
                            m.Pronoun_Id,
                            ISNULL(p.PronounName, 'Unknown') AS PronounName,
                            m.Sexual_Orient_Id,
                            ISNULL(so.SexualOrientationName, 'Unknown') AS SexualOrientationName,
                            m.StateId,
                            ISNULL(s.StateName, 'Unknown') AS StateName,
                            ISNULL(ct.CriticalThinkingNames, 'Unknown') AS CriticalThinkingNames,
                            ISNULL(gm.GrowthMindsetNames, 'Unknown') AS GrowthMindsetNames,
                            ISNULL(coll.CollaborationNames, 'Unknown') AS CollaborationNames,
                            ISNULL(mc.MetacognitionNames, 'Unknown') AS MetacognitionNames,
                            ISNULL(ch.CharacterNames, 'Unknown') AS CharacterNames,
                            ISNULL(cr.CreativityNames, 'Unknown') AS CreativityNames,
                            ISNULL(ft.FortitudeNames, 'Unknown') AS FortitudeNames,
                            ISNULL(ts.TechnicalSkillNames, 'Unknown') AS TechnicalSkillNames
                        FROM Member AS m
                        LEFT JOIN Age AS a ON m.Age_Id = a.Id
                        LEFT JOIN City AS c ON m.CityId = c.Id
                        LEFT JOIN Communication AS comm ON m.Communication_Id = comm.Id
                        LEFT JOIN EmploymentType AS et ON m.Employment_Type_Id = et.Id
                        LEFT JOIN Industry AS i ON m.Industry_Id = i.Id
                        LEFT JOIN JobLevel AS jl ON m.Job_Level_Id = jl.Id
                        LEFT JOIN Leadership AS l ON m.Leadership_Id = l.Id
                        LEFT JOIN MemberType AS mt ON m.MemberTypeId = mt.Id
                        LEFT JOIN Nationality AS n ON m.Nationality_Id = n.Id
                        LEFT JOIN Pronoun AS p ON m.Pronoun_Id = p.Id
                        LEFT JOIN SexualOrientation AS so ON m.Sexual_Orient_Id = so.Id
                        LEFT JOIN State AS s ON m.StateId = s.Id
                        LEFT JOIN Ethnicity AS Eth ON m.MemberEthnicityId = Eth.Id 
                        LEFT JOIN (
                            SELECT mct.MemberId, STRING_AGG(ct.CriticalThinkingName, ', ') AS CriticalThinkingNames
                            FROM Member_Critical_Thinkings AS mct
                            JOIN CriticalThinking AS ct ON mct.Critical_ThinkingId = ct.Id
                            GROUP BY mct.MemberId
                        ) AS ct ON m.Id = ct.MemberId
                        LEFT JOIN (
                            SELECT mgm.MemberId, STRING_AGG(gm.GrowthMindsetName, ', ') AS GrowthMindsetNames
                            FROM Member_Growth_Mindsets AS mgm
                            JOIN GrowthMindset AS gm ON mgm.Growth_MindsetId = gm.Id
                            GROUP BY mgm.MemberId
                        ) AS gm ON m.Id = gm.MemberId
                        LEFT JOIN (
                            SELECT mc.MemberId, STRING_AGG(c.CollaborationName, ', ') AS CollaborationNames
                            FROM Member_Collaborations AS mc
                            JOIN Collaboration AS c ON mc.CollaborationId = c.Id
                            GROUP BY mc.MemberId
                        ) AS coll ON m.Id = coll.MemberId
                        LEFT JOIN (
                            SELECT mm.MemberId, STRING_AGG(mc.MetacognitionName, ', ') AS MetacognitionNames
                            FROM Member_Metacognitions AS mm
                            JOIN Metacognition AS mc ON mm.MetacognitionId = mc.Id
                            GROUP BY mm.MemberId
                        ) AS mc ON m.Id = mc.MemberId
                        LEFT JOIN (
                            SELECT mchr.MemberId, STRING_AGG(ch.CharacterName, ', ') AS CharacterNames
                            FROM Member_Characters AS mchr
                            JOIN Character AS ch ON mchr.CharacterId = ch.Id
                            GROUP BY mchr.MemberId
                        ) AS ch ON m.Id = ch.MemberId
                        LEFT JOIN (
                            SELECT mc.MemberId, STRING_AGG(cr.CreativityName, ', ') AS CreativityNames
                            FROM Member_Creativitys AS mc
                            JOIN Creativity AS cr ON mc.CreativityId = cr.Id
                            GROUP BY mc.MemberId
                        ) AS cr ON m.Id = cr.MemberId
                        LEFT JOIN (
                            SELECT mf.MemberId, STRING_AGG(ft.FortitudeName, ', ') AS FortitudeNames
                            FROM Member_Fortitudes AS mf
                            JOIN Fortitude AS ft ON mf.FortitudeId = ft.Id
                            GROUP BY mf.MemberId
                        ) AS ft ON m.Id = ft.MemberId
                        LEFT JOIN (
                            SELECT mts.MemberId, STRING_AGG(ts.TechnicalSkillsName, ', ') AS TechnicalSkillNames
                            FROM Member_TechnicalSkills AS mts
                            JOIN TechnicalSkills AS ts ON mts.TechnicalSkillId = ts.Id
                            GROUP BY mts.MemberId
                        ) AS ts ON m.Id = ts.MemberId;
                    """
                }

                table_details = {}

                for table, query in queries.items():
                    try:
                        if table == "jobpost":
                            cursor.execute(query, job_id)
                        else:
                            cursor.execute(query)
                        columns = [column[0] for column in cursor.description]
                        rows = []
                        for row in cursor.fetchall():
                            row_dict = dict(zip(columns, row))
                            for key, value in row_dict.items():
                                if isinstance(value, datetime):
                                    row_dict[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                                elif isinstance(value, (bytes, bytearray)):
                                    row_dict[key] = value.decode('utf-8', errors='ignore')
                                elif isinstance(value, Decimal):
                                    row_dict[key] = float(value)
                            rows.append(row_dict)

                        table_details[table] = {
                            "full_data": rows
                        }

                        # Logging the number of records fetched
                        print(f"Fetched {len(rows)} records from '{table}' table.")

                    except Exception as e:
                        print(f"Error fetching data for {table}: {e}")
                        return {'error': f'Error fetching data for {table}'}

                cursor.close()
                conn.close()

                return table_details
            except Exception as e:
                print(f"Database connection error: {e}")
                return {'error': 'Database connection error'}

        # Fetch data asynchronously
        loop = asyncio.get_event_loop()
        table_details = await loop.run_in_executor(None, fetch_data_sync)
        if isinstance(table_details, dict) and 'error' in table_details:
            return table_details

        # Extract job posts and members
        job_posts = table_details.get("jobpost", {}).get("full_data", [])
        members = table_details.get("member", {}).get("full_data", [])

        if not job_posts or not members:
            return {'error': 'Data fetching error'}

        # Create textual representations (keep your existing functions)
        job_texts = [create_job_text(job) for job in job_posts if create_job_text(job)]
        member_texts = [create_member_text(member) for member in members if create_member_text(member)]

        if not job_texts:
            return {'error': 'No valid job texts to process.'}
        if not member_texts:
            return {'error': 'No valid member texts to process.'}

        # Generate embeddings
        job_embeddings = [get_bert_embedding(text) for text in job_texts]
        member_embeddings = [get_bert_embedding(text) for text in member_texts]

        # Match jobs with members
        job_to_member_matches = []

        for job_idx, job in enumerate(job_posts):
            job_embedding = job_embeddings[job_idx]
            similarities = [calculate_similarity(job_embedding, mem_emb) for mem_emb in member_embeddings]
            
            # Convert similarities to percentages and filter
            similarities_percent = [sim * 100 for sim in similarities]
            # filtered = [(i, sim) for i, sim in enumerate(similarities_percent) if sim >= 80]  # 70% threshold
            filtered = [(i, sim) for i, sim in enumerate(similarities_percent) if sim >= 60]  # Lower threshold to 60%
            filtered.sort(key=lambda x: x[1], reverse=True)
            
            job_details = get_job_post_details(job)
            matched_members = []
            # matched_members_dict = {}

            
            for rank, (member_idx, sim_percent) in enumerate(filtered[:150], start=1):  # Top 150 matches
                member = members[member_idx]
                member_details = get_member_details(member)
                member_details["Rank"] = rank
                member_details["Similarity"] = f"{sim_percent:.2f}"
                
                # Call explain_similarity function asynchronously
                member_details = await explain_similarity(job_details, member_details, openai_api_key)
                
                # Append member_details only once
                matched_members.append(member_details)

            job_to_member_matches.append({
                "Job Details": job_details,
                "Matched Members": matched_members
            })

            return job_to_member_matches

    except Exception as e:
        print(f"Unexpected error: {e}")
        return {'error': 'An unexpected error occurred.'}


@app.route('/get_recommended_profiles/<int:job_id>', methods=['GET'])
async def get_recommended_profiles(job_id):
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        recommendations = await process_recommendations(job_id, openai_api_key)
        if isinstance(recommendations, dict) and 'error' in recommendations:
            return jsonify(recommendations), 500

        return jsonify(recommendations)

    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500

if __name__ == "__main__":
    app.run(host='127.0.0.3', port=5000, debug=False)