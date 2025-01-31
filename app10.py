from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
from decimal import Decimal
import pyodbc
from flask import Flask, jsonify
from functools import wraps
import asyncio
from pyngrok import ngrok

app = Flask(__name__)
CORS(app,supports_credentials=True)
ngrok.set_auth_token("2sLHDndoqMyvZJrDP7TwfbvHLBg_5cvZ1d9YNVWqoeYH7LCd5")
public_url = ngrok.connect(5000).public_url  # Assumes Flask runs on port 5000
print("Public URL:", public_url)


#Database 

def async_route(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapped

async def fetch_data_async(job_id):
    try:
        conn = pyodbc.connect(
            'DRIVER={SQL Server};'
            'SERVER=DESKTOP-68FBO4B;'
            'DATABASE=MPOWER_TEST;'
            'Trusted_Connection=yes;'
        )
        cursor = conn.cursor()

        # Your existing queries dictionary remains the same
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
                            jp.Job_Description AS Description,
                            jp.Qualifications,
                            jp.PreferredSkills,
                            jp.Required_Skills,
                            jp.Key_Responsibilities,
                            ISNULL(ind.IndustryName, 'Unknown') AS Industry,
                            jp.JobTitle AS Role,
                            jp.IsActive, 
                            jp.CreatedDate, 
                            jp.ModifiedDate
                        FROM JobPost AS jp
                        LEFT JOIN EmploymentType AS et ON jp.EmploymentTypeId = et.Id
                        LEFT JOIN City AS c ON jp.CityId = c.Id
                        LEFT JOIN Industry AS ind ON jp.IndustryId = ind.Id
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
                                Eth.EthnicityName,
                                m.Headline,
                                ISNULL(comm.CommunicationNames, 'Unknown') AS CommunicationNames,
                                ISNULL(comm.CommunicationDescriptions, 'Unknown') AS CommunicationDescriptions,
                                ISNULL(l.LeadershipNames, 'Unknown') AS LeadershipNames,
                                ISNULL(l.LeadershipDescriptions, 'Unknown') AS LeadershipDescriptions,
                                ISNULL(ct.CriticalThinkingNames, 'Unknown') AS CriticalThinkingNames,
                                ISNULL(ct.CriticalThinkingDescriptions, 'Unknown') AS CriticalThinkingDescriptions,
                                ISNULL(collab.CollaborationNames, 'Unknown') AS CollaborationNames,
                                ISNULL(collab.CollaborationDescriptions, 'Unknown') AS CollaborationDescriptions,
                                ISNULL(ch.CharacterNames, 'Unknown') AS CharacterNames,
                                ISNULL(ch.CharacterDescriptions, 'Unknown') AS CharacterDescriptions,
                                ISNULL(cr.CreativityNames, 'Unknown') AS CreativityNames,
                                ISNULL(cr.CreativityDescriptions, 'Unknown') AS CreativityDescriptions,
                                ISNULL(gm.GrowthMindsetNames, 'Unknown') AS GrowthMindsetNames,
                                ISNULL(gm.GrowthMindsetDescriptions, 'Unknown') AS GrowthMindsetDescriptions,
                                ISNULL(mind.MindfulnessNames, 'Unknown') AS MindfulnessNames,
                                ISNULL(mind.MindfulnessDescriptions, 'Unknown') AS MindfulnessDescriptions,
                                ISNULL(f.FortitudeNames, 'Unknown') AS FortitudeNames,
                                ISNULL(f.FortitudeDescriptions, 'Unknown') AS FortitudeDescriptions,
                                ISNULL(ts.TechnicalSkillNames, 'Unknown') AS TechnicalSkillNames,
                                ISNULL(ts.TechnicalSkillDescriptions, 'Unknown') AS TechnicalSkillDescriptions,
                                ISNULL(me.EducationInfo, 'Unknown') AS Education,
                                ISNULL(mex.ExperienceInfo, 'Unknown') AS Experience,
                                ISNULL(mos.OtherSkillNames, 'Unknown') AS OtherSkills
                            FROM Member AS m
                            LEFT JOIN Age AS a ON m.Age_Id = a.Id
                            LEFT JOIN City AS c ON m.CityId = c.Id
                            LEFT JOIN Ethnicity AS Eth ON m.MemberEthnicityId = Eth.Id
                            LEFT JOIN (
                                SELECT MemberId, 
                                    STRING_AGG(CommunicationName, '; ') AS CommunicationNames,
                                    STRING_AGG(CommunicationDescription, '; ') AS CommunicationDescriptions
                                FROM Member_Communications mc
                                JOIN Communication comm ON mc.CommunicationId = comm.Id
                                GROUP BY MemberId
                            ) AS comm ON m.Id = comm.MemberId
                            LEFT JOIN (
                                SELECT MemberId, 
                                    STRING_AGG(LeadershipName, '; ') AS LeadershipNames,
                                    STRING_AGG(LeadershipDescription, '; ') AS LeadershipDescriptions
                                FROM Member_Leaderships ml
                                JOIN Leadership l ON ml.LeadershipId = l.Id
                                GROUP BY MemberId
                            ) AS l ON m.Id = l.MemberId
                            LEFT JOIN (
                                SELECT MemberId, 
                                    STRING_AGG(CriticalThinkingName, '; ') AS CriticalThinkingNames,
                                    STRING_AGG(CriticalThinkingDescription, '; ') AS CriticalThinkingDescriptions
                                FROM Member_Critical_Thinkings mct
                                JOIN CriticalThinking ct ON mct.Critical_ThinkingId = ct.Id
                                GROUP BY MemberId
                            ) AS ct ON m.Id = ct.MemberId
                            LEFT JOIN (
                                SELECT MemberId, 
                                    STRING_AGG(CollaborationName, '; ') AS CollaborationNames,
                                    STRING_AGG(CollaborationDescription, '; ') AS CollaborationDescriptions
                                FROM Member_Collaborations mcol
                                JOIN Collaboration collab ON mcol.CollaborationId = collab.Id
                                GROUP BY MemberId
                            ) AS collab ON m.Id = collab.MemberId
                            LEFT JOIN (
                                SELECT MemberId, 
                                    STRING_AGG(CharacterName, '; ') AS CharacterNames,
                                    STRING_AGG(CharacterDescription, '; ') AS CharacterDescriptions
                                FROM Member_Characters mch
                                JOIN Character ch ON mch.CharacterId = ch.Id
                                GROUP BY MemberId
                            ) AS ch ON m.Id = ch.MemberId
                            LEFT JOIN (
                                SELECT MemberId, 
                                    STRING_AGG(CreativityName, '; ') AS CreativityNames,
                                    STRING_AGG(CreativityDescription, '; ') AS CreativityDescriptions
                                FROM Member_Creativitys mcr
                                JOIN Creativity cr ON mcr.CreativityId = cr.Id
                                GROUP BY MemberId
                            ) AS cr ON m.Id = cr.MemberId
                            LEFT JOIN (
                                SELECT MemberId, 
                                    STRING_AGG(GrowthMindsetName, '; ') AS GrowthMindsetNames,
                                    STRING_AGG(GrowthMindsetDescription, '; ') AS GrowthMindsetDescriptions
                                FROM Member_Growth_Mindsets mgm
                                JOIN GrowthMindset gm ON mgm.Growth_MindsetId = gm.Id
                                GROUP BY MemberId
                            ) AS gm ON m.Id = gm.MemberId
                            LEFT JOIN (
                                SELECT MemberId, 
                                    STRING_AGG(MindfulnessName, '; ') AS MindfulnessNames,
                                    STRING_AGG(MindfulnessDescription, '; ') AS MindfulnessDescriptions
                                FROM Member_Mindfulness mm
                                JOIN Mindfulness mind ON mm.MindfulnessId = mind.Id
                                GROUP BY MemberId
                            ) AS mind ON m.Id = mind.MemberId
                            LEFT JOIN (
                                SELECT MemberId, 
                                    STRING_AGG(FortitudeName, '; ') AS FortitudeNames,
                                    STRING_AGG(FortitudeDescription, '; ') AS FortitudeDescriptions
                                FROM Member_Fortitudes mf
                                JOIN Fortitude f ON mf.FortitudeId = f.Id
                                GROUP BY MemberId
                            ) AS f ON m.Id = f.MemberId
                            LEFT JOIN (
                                SELECT MemberId, 
                                    STRING_AGG(TechnicalSkillsName, '; ') AS TechnicalSkillNames,
                                    STRING_AGG(TechnicalSkillsDescription, '; ') AS TechnicalSkillDescriptions
                                FROM Member_TechnicalSkills mts
                                JOIN TechnicalSkills ts ON mts.TechnicalSkillId = ts.Id
                                GROUP BY MemberId
                            ) AS ts ON m.Id = ts.MemberId
                            LEFT JOIN (
                                SELECT MemberId, STRING_AGG(EducationDegree + ' - ' + EducationFieldStudy + ' - ' + EducationDescription, '; ') AS EducationInfo
                                FROM MemberEducation
                                GROUP BY MemberId
                            ) AS me ON m.Id = me.MemberId
                            LEFT JOIN (
                                SELECT MemberId, STRING_AGG(ExperienceJobTitle + ' at ' + ExperienceCompany + ' - ' + ExperienceDescription, '; ') AS ExperienceInfo
                                FROM MemberExperience
                                GROUP BY MemberId
                            ) AS mex ON m.Id = mex.MemberId
                            LEFT JOIN (
                                SELECT MemberId, STRING_AGG(OtherSkillName, ', ') AS OtherSkillNames
                                FROM Member_OtherSkills
                                GROUP BY MemberId
                            ) AS mos ON m.Id = mos.MemberId;
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
                    row_dict = {}
                    for idx, value in enumerate(row):
                        if isinstance(value, datetime):
                            row_dict[columns[idx]] = value.strftime('%Y-%m-%d %H:%M:%S')
                        elif isinstance(value, (bytes, bytearray)):
                            row_dict[columns[idx]] = value.decode('utf-8', errors='ignore')
                        elif isinstance(value, Decimal):
                            row_dict[columns[idx]] = float(value)
                        else:
                            row_dict[columns[idx]] = value
                    rows.append(row_dict)

                table_details[table] = rows  # Changed to directly store the rows
                print(f"Fetched {len(rows)} records from '{table}' table.")

            except Exception as e:
                print(f"Error fetching data for {table}: {e}")
                return {'error': f'Error fetching data for {table}'}

        cursor.close()
        conn.close()

        if not table_details:
            return {'error': 'No data found'}
            
        return table_details

    except Exception as e:
        print(f"Database connection error: {e}")
        return {'error': 'Database connection error'}





# Machine Learing model logic implementation starts here
# Load BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def preprocess_text(text):
    return ' '.join(str(text).lower().split())

def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def calculate_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

def match_skills(member_skills, job_skills, required=False):
    score = 0
    if job_skills:
        for skill in job_skills.split(','):
            if skill.lower() in member_skills.lower():
                score += 3 if required else 1
            elif any(s in skill.lower() for s in member_skills.lower().split()):
                score += 2 if required else 1
    return score


def match_profile(member, job):
    member_city = member.get('CityName', '') or ''
    job_location = job.get('JobLocation', '') or ''
    member_headline = member.get('Headline', '') or ''
    job_role = job.get('Role', '') or ''
    
    scores = {
        'required_skills': match_skills(member.get('TechnicalSkillNames', ''), job.get('Required_Skills', '') or '', required=True),
        'preferred_skills': match_skills(member.get('TechnicalSkillNames', ''), job.get('PreferredSkills', '') or ''),
        'other_skills': match_skills(member.get('OtherSkills', ''), (job.get('Required_Skills', '') or '') + (job.get('PreferredSkills', '') or '')),
        'qualifications': calculate_similarity(encode_text(member.get('Education', '')), encode_text(job.get('Qualifications', ''))),
        'responsibilities': calculate_similarity(encode_text(member.get('Experience', '')), encode_text(job.get('Key_Responsibilities', ''))),
        'industry': 3 if member.get('Industry', '').lower() == job.get('Industry', '').lower() else 2 if member.get('Industry', '').lower() in job.get('Industry', '').lower() else 0,
        'role': 3 if member_headline.lower() == job_role.lower() else 
                2 if job_role.lower() in member_headline.lower() else 0,
        'location': 2 if member_city.lower() == job_location.lower() else 1 if member_city.lower() in job_location.lower() else 0
    }
    
    weights = {
        'required_skills': 0.15,
        'preferred_skills': 0.15,
        'other_skills': 0.25,
        'qualifications': 0.15,
        'responsibilities': 0.15,
        'industry': 0.05,
        'role': 0.05,
        'location': 0.05
    }
    
    total_score = sum(scores[k] * weights[k] for k in scores) * 100
    result =min(total_score, 100)
    rounded = round(result, 2)
    formatted = f"{rounded:.2f}"
    return formatted # Convert to percentage


# def scale_to_30_100_rounded(old_value, old_min=0, old_max=50, decimals=0):
#     """
#     Scales a value (default original range: 0–50) to 30–100 and rounds the result.
#     """
#     new_min, new_max = 20, 100
    
#     if old_max == old_min:
#         raise ValueError("Original range cannot have zero width (old_min == old_max)")
    
#     scaled = ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
#     clamped = max(min(scaled, new_max), new_min)
#     return round(clamped, decimals)



def rank_candidates(members, job):
    rankings = {
        "Job Details": {
            "Job Id": job['Id'],
            "Job Title": job['JobTitle'],
            "Job Code": job['JobCode'],
            "City": job['CityName'],
            "Job Location": job['JobLocation'],
            "Employment Type": job['EmploymentTypeName'],
            "Posted Date": job['PostedDate'],
            "Deadline for Applications": job['Deadline_for_Applications'],
            "Is Active": job['IsActive'],
            "Salary Range Start": job['Salary_Range_Start'],
            "Salary Range End": job['Salary_Range_End']
        },
        "Matched Members": []
    }

    matched_candidates = []
    for member in members:
        match_percentage = float(match_profile(member, job))
        # match_percentage_scaled = scale_to_30_100_rounded(match_percentage)
        match_percentage_scaled = match_percentage
        
        if match_percentage_scaled >= 85:
            matched_candidates.append({
                "Member Id": member['MemberId'],
                "Member First Name": member['MemberFirstName'],
                "Member Last Name": member['MemberLastName'],
                "Member Email": member['MemberEmail'],
                "City": member.get('CityName'),
                "Similarity": match_percentage_scaled,
                
            })

    rankings["Matched Members"] = sorted(matched_candidates, 
                                       key=lambda x: float(x["Similarity"]), 
                                       reverse=True)
    return rankings


# @app.route('/get_recommended_profiles', methods=['POST'])
# def match_job():
#     data = request.json
#     job_posting = data['data']['jobpost'][0]
#     candidates = data['data']['member']
    
#     rankings = rank_candidates(candidates, job_posting)
#     return jsonify(rankings)

@app.route('/get_recommended_profiles', methods=['POST'])
def match_job():
    # Use force=True to parse JSON even if Content-Type isn't application/json
    data = request.get_json(force=True)  
    
    if not data or 'data' not in data or 'jobpost' not in data['data'] or 'member' not in data['data']:
        return jsonify({"error": "Invalid data format"}), 400
    
    job_posting = data['data']['jobpost'][0]
    candidates = data['data']['member']
    
    rankings = rank_candidates(candidates, job_posting)
    return jsonify(rankings)



@app.route('/get_profiles_data/<int:job_id>', methods=['GET'])
@async_route
async def get_recommended_profiles(job_id):
    try:
        recommendations = await fetch_data_async(job_id)
        
        if isinstance(recommendations, dict) and 'error' in recommendations:
            return jsonify(recommendations), 500
            
        # Ensure the response is JSON serializable
        return jsonify({
            'status': 'success',
            'data': recommendations
        })

    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == "__main__":
    app.run()