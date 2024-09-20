from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
import streamlit as st
import PyPDF2
import base64
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables (e.g., OpenAI API key)
load_dotenv()

# Initialize the language model (replace with your preferred model)
llm = ChatOpenAI(model="gpt-4o-mini")  # Or gpt-4 if available

# --- Pydantic Models for Data Structures ---
class PersonalDetails(BaseModel):
    name: str = Field(description="Name of the person")
    email: str = Field(description="Email id of the person")
    contact_num: str = Field(description="Phone Number of the person")

class Education(BaseModel):
    university: str = Field(description="Name of the university")
    degree: str = Field(description="Degree obtained")
    year_of_passing: Optional[str] = Field(description="Year of passing")
    field_of_study: Optional[str] = Field(description="Field of study")
    grade: Optional[str] = Field(description="Grade obtained")

class Project(BaseModel):
    project_name: str = Field(description="Title of the project")
    description: str = Field(description="Description of the project")

class Skill(BaseModel):
    skill_name: str = Field(description="Name of the skill")
    proficiency_level: Optional[str] = Field(description="Proficiency level of the skill")

class WorkTask(BaseModel):
    task: str = Field(description="Task performed at the job")

class Experience(BaseModel):
    company_name: str = Field(description="Name of the company")
    job_role: str = Field(description="Job role at the company")
    duration: str = Field(description="Duration of the job")
    tasks: List[WorkTask] = Field(description="List of tasks performed at the job")

class Resume(BaseModel):
    personal_details: PersonalDetails
    education: List[Education]
    experience: List[Experience]
    skills: List[Skill]
    projects: List[Project]

class ResumeScores(BaseModel):
    experience_score: int = Field(..., ge=1, le=10)
    experience_feedback: str
    education_score: int = Field(..., ge=1, le=10)
    education_feedback: str
    skills_score: int = Field(..., ge=1, le=10)
    skills_feedback: str
    projects_score: int = Field(..., ge=1, le=10)
    projects_feedback: str
    overall_score: int = Field(..., ge=1, le=10)
    overall_feedback: str

class Suggestion(BaseModel):
    original_task: List[str] = Field(description="List of original work tasks mentioned in experience.")
    reframed: List[str] = Field(description="List of corresponding reframed work tasks")

# --- Helper Functions ---
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def create_chart_overall(value: int):
    fig, ax = plt.subplots(figsize=(3, 3))
    value = value * 10
    sizes = [100 - value, value]
    colors = ['silver', 'blue']
    explode = (0, 0.1)
    ax.pie(sizes, explode=explode, colors=colors, startangle=90, shadow=True)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)
    ax.text(0, 0, f'{value}/100', ha='center', va='center')
    ax.axis('equal')
    return fig

def create_chart(value: int):
    fig, ax = plt.subplots()
    sizes = [10 - value, value]
    ax.pie(sizes, colors=['red', 'green'], startangle=90)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)
    ax.text(0, 0, f'{value}/10', ha='center', va='center')
    return fig

def extract_info(resume: str):
    parser = OutputFixingParser.from_llm(parser=PydanticOutputParser(pydantic_object=Resume), llm=llm)
    format_instructions = parser.get_format_instructions()
    resume_text = llm.predict(
        f"Given a resume {resume} \n Extract all the relevant sections.  \n {format_instructions}")
    resume_info = parser.parse(resume_text)
    return resume_info

def description_evaluation(resume, job_description):
    prompt_template = f"""You are an Resume Expert. Your job is to give feedback on the resume based on the provided job description.
    Be specific about the points.
    
    Resume: {resume}
    
    Job Description: {job_description}
    
    Please provide the feedback in the following format.
    
    ## Strengths:
    <list strengths here>
    
    ## Weaknesses:
    <list weaknesses here>
    
    ## Recommendations to improve CV:
    <list recommendations here>
    
    ONLY QUOTE THE INFORMATION PROVIDED IN THE RESUME. DO NOT MAKE UP INFORMATION WHICH IS NOT EXPLICITLY PROVIDED IN RESUME.
    RETURN THE RESPONSE IN MARKDOWN FORMAT IN BULLET POINTS.
    """
    output = llm.predict(prompt_template)  # Use llm.predict() here

    # Parse the output to extract the sections and format them
    try:
        strengths_start = output.index("## Strengths:")
        weaknesses_start = output.index("## Weaknesses:")
        recommendations_start = output.index("## Recommendations to improve CV:")

        strengths = output[strengths_start + len("## Strengths:"):weaknesses_start].strip().split("- ")
        weaknesses = output[weaknesses_start + len("## Weaknesses:"):recommendations_start].strip().split("- ")
        recommendations = output[recommendations_start + len("## Recommendations to improve CV:"):].strip().split("- ")

        # Format the output
        formatted_output = ""
        if strengths:
            formatted_output += "**Strengths:**\n"
            for strength in strengths:
                formatted_output += f"- {strength}\n"
        if weaknesses:
            formatted_output += "\n**Weaknesses:**\n"
            for weakness in weaknesses:
                formatted_output += f"- {weakness}\n"
        if recommendations:
            formatted_output += "\n**Recommendations to improve CV:**\n"
            for recommendation in recommendations:
                formatted_output += f"- {recommendation}\n"
        return formatted_output
    except ValueError:
        return "Could not parse the feedback. Please check the prompt and try again."

def llm_scoring(llm, resume_text, job_description):
    prompt = f"""
    Given the following resume for the job role '{job_description}', please evaluate and provide a score between 1 to 10 (where 1 is the lowest and 10 is the highest), and provide feedback for each category and the overall resume:

    {resume_text}

    Categories:
    1. Relevant Experience
    2. Education
    3. Skills
    4. Projects

    Here are some rules for the scores:
    - Provide honest scores based on the resume. 
    - Give higher scores (8, 9, 10) only in rare cases.
    - Relevant Experience should be high only when the current job is the same as the applied job role.
    - Education Experience should be high only when the candidate is from a premier college.
    - Skills and Projects should be evaluated in conjunction with the applied role. Give a low score (<6) if there are no relevant projects.
    - The score should be integers between 1 to 10. 

    Take a deep breath. Read the above instructions clearly before giving the scores.

    Relevant Experience: {{score_experience}}, Feedback: {{feedback_experience}}
    Education: {{score_education}}, Feedback: {{feedback_education}}
    Skills: {{score_skills}}, Feedback: {{feedback_skills}}
    Projects: {{score_projects}}, Feedback: {{feedback_projects}}
    Overall Score: {{score_overall}}, Feedback: {{feedback_overall}}
    """
    response = llm.predict(prompt)

    parser = OutputFixingParser.from_llm(parser=PydanticOutputParser(pydantic_object=ResumeScores), llm=llm)
    resume_scores = parser.parse(response)

    return resume_scores


def suggest_improvements(llm, experience):
    prompt = f"""
    Given the following resume for the job role, please evaluate and provide improvements to the work tasks using the below hints:
    HINTS: Quantification of work, use of strong action words, overall impact made.

    {experience}

    Select any 4 to 10 work tasks and reframe them for better results.
    """
    response = llm.predict(prompt)

    parser = OutputFixingParser.from_llm(parser=PydanticOutputParser(pydantic_object=Suggestion), llm=llm)
    suggestions = parser.parse(response)

    return suggestions

# --- Streamlit App ---
def main():
    st.set_page_config(layout="wide")
    st.title("Welcome to Resumoid 🤖")
    st.subheader("🌝 Your personal AI ATS!")

    st.markdown("📄 Upload your resume and job role to get feedback in 2 minutes!")

    resume_pdf = st.file_uploader("Upload your resume", type=['pdf'], label_visibility='collapsed')
    job_description = st.text_input("Enter the role for which you are applying")

    submit = st.button("Submit")

    if resume_pdf and job_description and submit:
        resume_text = read_pdf(resume_pdf)
        resume_info = extract_info(resume_text)
        gpt4_model = ChatOpenAI(model='gpt-4')  # or gpt-3.5-turbo-16k
        resume_scores = llm_scoring(llm=gpt4_model, resume_text=resume_text, job_description=job_description)

        st.divider()

        st.markdown("### Candidate Details")

        st.markdown("**Name:** " + resume_info.personal_details.name)
        st.markdown("**Email:** " + resume_info.personal_details.email)
        st.markdown("**Contact Number:** " + resume_info.personal_details.contact_num)
        st.markdown("**University:** " + resume_info.education[0].university)
        st.markdown("**Current Job Role:** " + resume_info.experience[0].company_name)
        st.markdown("**Company:** " + resume_info.experience[0].job_role)

        st.divider()

        ocol1, ocol2, ocol3 = st.columns(3)

        ocol2.markdown("### Relevance Score \n\n\n\n")
        ocol2.pyplot(create_chart_overall(resume_scores.overall_score))
        ocol2.markdown(resume_scores.overall_feedback)

        st.divider()

        st.markdown("### Evaluation")

        st.text(f"Here is the evaluation of your resume for the {job_description} role.")

        col1, col2, col3, col4 = st.columns(4)
        # Column 1
        col1.markdown("### Experience \n\n\n")
        col1.pyplot(create_chart(resume_scores.experience_score))
        col1.markdown(resume_scores.experience_feedback)

        # Column 2
        col2.markdown("### Education \n\n\n")
        col2.pyplot(create_chart(resume_scores.education_score))
        col2.markdown(resume_scores.education_feedback)

        # Column 3
        col3.markdown("### Skills \n\n\n\n")
        col3.pyplot(create_chart(resume_scores.skills_score))
        col3.markdown(resume_scores.skills_feedback)

        # Column 4
        col4.markdown("### Projects \n\n\n\n")
        col4.pyplot(create_chart(resume_scores.projects_score))
        col4.markdown(resume_scores.projects_feedback)

        st.divider()

        st.markdown("### Detailed Comments")
        feedback_jobdesc = description_evaluation(resume_text, job_description)
        st.markdown(feedback_jobdesc)  # Display the formatted output here

        st.markdown("### Suggestions")
        output = suggest_improvements(llm, resume_info.experience)

        original_tasks = output.original_task
        improvised_tasks = output.reframed

        col4, col5 = st.columns(2)
        col4.markdown("#### Your Points")
        col5.markdown("#### Suggested Improvement")

        for task, suggestion in zip(original_tasks, improvised_tasks):
            x1, x2 = st.columns(2)
            x1.markdown(f"- :red[{task}]")
            x2.markdown(f"- :green[{suggestion}]")
            st.markdown("---------------")

        st.divider()

        st.success(""" Chat feature coming soon! \n
        Reach out to me at satvik@buildfastwithai.com""")

if __name__ == '__main__':
    main() 
