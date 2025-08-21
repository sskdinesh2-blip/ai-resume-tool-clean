from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import fitz as PyMuPDF  # for PDF processing
import docx    # for Word document processing
import io
import os
import spacy
import re
from textblob import TextBlob
from typing import List, Dict
import json
from typing import Optional
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="AI Resume Customizer API",
    description="Backend API for AI-powered resume customization",
    version="2.0.0"
)

# Enable CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
 "http://localhost:3000",
        "https://effervescent-gaufre-c465b9.netlify.app"
            ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
# For now, we'll set the API key directly in code (you'll move this to environment variable later)
openai_client = OpenAI(
    api_key="NEW OPEN API KEY HERE"  # Replace with your real key
)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

# Root endpoint
@app.get("/")
async def root():
    return {"message": "AI Resume Customizer API v2.0 with GPT integration! 🚀🤖"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API with GPT integration is working perfectly!"}

async def safe_openai_call(prompt, max_tokens=1000, temperature=0.3):
    """Wrapper for OpenAI calls with error handling"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert resume advisor."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=30
        )
        return {"success": True, "content": response.choices[0].message.content}
        
    except Exception as e:
        error_type = type(e).__name__
        if "authentication" in str(e).lower():
            return {
                "success": False,
                "error": "authentication_error",
                "user_message": "AI features temporarily unavailable. Using standard optimization.",
                "fallback_available": True
            }
        elif "rate_limit" in str(e).lower():
            return {
                "success": False,
                "error": "rate_limit_error",
                "user_message": "AI service is busy. Please wait and try again.",
                "retry_after": 60
            }
        elif "timeout" in str(e).lower():
            return {
                "success": False,
                "error": "timeout_error", 
                "user_message": "AI processing timed out. Using fallback optimization.",
                "fallback_available": True
            }
        else:
            return {
                "success": False,
                "error": "general_error",
                "user_message": "AI processing unavailable. Using standard optimization.",
                "fallback_available": True
            }

# Resume upload and parsing endpoint
@app.post("/api/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    try:
        # File size validation
        if file.size and file.size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "File too large",
                    "message": f"File size {file.size / (1024*1024):.1f}MB exceeds 10MB limit",
                    "suggestion": "Please compress your file or use a smaller version"
                }
            )
        
        # File type validation
        valid_types = [
            "application/pdf",
            "application/msword", 
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]
        
        if file.content_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid file type",
                    "received": file.content_type,
                    "allowed": ["PDF", "DOC", "DOCX"],
                    "suggestion": "Please convert your resume to PDF, DOC, or DOCX format"
                }
            )
        
        # Read file content
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Empty file",
                    "message": "The uploaded file appears to be empty"
                }
            )
            
        extracted_text = ""
        
        if file.content_type == "application/pdf":
            try:
                pdf_document = PyMuPDF.open(stream=file_content, filetype="pdf")
                for page_num in range(pdf_document.page_count):
                    page = pdf_document[page_num]
                    extracted_text += page.get_text()
                pdf_document.close()
            except Exception as pdf_error:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "PDF processing failed",
                        "message": str(pdf_error),
                        "suggestion": "Try saving your PDF in a different format"
                    }
                )
        elif file.content_type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            # Process Word document
            try:
                doc = docx.Document(io.BytesIO(file_content))
                for paragraph in doc.paragraphs:
                    extracted_text += paragraph.text + "\n"
            except Exception as docx_error:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "DOCX processing failed",
                        "message": str(docx_error),
                        "suggestion": "Try saving your document in a different format"
                    }
                )
        
        if len(extracted_text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Insufficient content",
                    "message": f"Only {len(extracted_text.strip())} characters extracted",
                    "suggestion": "Your resume might be image-based. Try uploading a text-based version"
                }
            )
        
        # Return success response with extracted text
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Resume uploaded and processed successfully!",
                "filename": file.filename,
                "file_size": len(file_content),
                "extracted_text": extracted_text,
                "full_text_length": len(extracted_text),
                "metadata": {
                    "content_type": file.content_type,
                    "original_filename": file.filename
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Processing failed",
                "message": "An unexpected error occurred",
                "suggestion": "Please try again or contact support"
            }
        )

# Job description analysis endpoint (enhanced)
@app.post("/api/analyze-job")
async def analyze_job_description(job_description: str = Form(...)):
    """
    Enhanced AI-powered job description analysis with advanced NLP
    """
    try:
        # Enhanced analysis using GPT with better prompting
        enhanced_analysis_prompt = f"""
        As an expert HR analyst and recruitment specialist, perform a comprehensive analysis of this job description:

        Job Description:
        {job_description}

        Provide a detailed JSON analysis with these exact keys:
        {{
            "company_analysis": {{
                "company_size": "startup/small/medium/large/enterprise",
                "industry": "specific industry name",
                "company_culture": ["collaborative", "fast-paced", "innovative", etc.],
                "remote_policy": "remote/hybrid/onsite/flexible"
            }},
            "role_analysis": {{
                "seniority_level": "entry/junior/mid/senior/principal/executive",
                "role_type": "individual_contributor/team_lead/manager/director",
                "department": "engineering/data/product/marketing/etc"
            }},
            "skills_breakdown": {{
                "required_technical_skills": ["Python", "SQL", etc.],
                "required_soft_skills": ["communication", "leadership", etc.],
                "preferred_technical_skills": ["AWS", "Docker", etc.],
                "preferred_soft_skills": ["mentoring", "public speaking", etc.],
                "nice_to_have": ["specific certifications", etc.]
            }},
            "experience_requirements": {{
                "years_required": "X-Y years or entry level",
                "education_required": "Bachelor's/Master's/PhD/bootcamp/none",
                "specific_experience": ["startups", "enterprise", "specific industries"]
            }},
            "compensation_indicators": {{
                "salary_range": "if mentioned or 'not specified'",
                "equity_mentioned": true/false,
                "benefits_highlighted": ["health", "401k", "pto", etc.]
            }},
            "application_insights": {{
                "urgency_level": "low/medium/high",
                "competition_level": "low/medium/high based on requirements",
                "key_differentiators": ["what makes candidates stand out"],
                "red_flags": ["any concerning aspects"]
            }},
            "optimization_keywords": {{
                "primary_keywords": ["most important 5-7 keywords"],
                "secondary_keywords": ["supporting keywords"],
                "ats_critical_terms": ["exact phrases that matter for ATS"]
            }}
        }}
        """

        # Call GPT for enhanced analysis
        gpt_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert HR analyst with 15+ years of experience in recruitment, ATS systems, and job market analysis. Provide detailed, actionable insights."},
                {"role": "user", "content": enhanced_analysis_prompt}
            ],
            max_tokens=1200,
            temperature=0.2
        )

        gpt_analysis_raw = gpt_response.choices[0].message.content
        
        # Try to parse JSON from GPT response
        try:
            # Extract JSON from GPT response (sometimes it includes extra text)
            json_match = re.search(r'\{.*\}', gpt_analysis_raw, re.DOTALL)
            if json_match:
                gpt_analysis = json.loads(json_match.group())
            else:
                gpt_analysis = {"error": "Could not parse GPT response", "raw": gpt_analysis_raw}
        except json.JSONDecodeError:
            gpt_analysis = {"error": "Invalid JSON from GPT", "raw": gpt_analysis_raw}

        # Enhanced NLP analysis using spaCy
        nlp_insights = {}
        if nlp:
            doc = nlp(job_description)
            
            # Extract entities
            entities = {
                "organizations": [ent.text for ent in doc.ents if ent.label_ == "ORG"],
                "technologies": [],
                "locations": [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]],
                "money": [ent.text for ent in doc.ents if ent.label_ == "MONEY"]
            }
            
            # Technical skills detection
            tech_keywords = [
                "Python", "JavaScript", "Java", "C++", "React", "Angular", "Vue",
                "SQL", "MongoDB", "PostgreSQL", "MySQL", "Redis",
                "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Jenkins",
                "Git", "GitHub", "GitLab", "Jira", "Confluence",
                "TensorFlow", "PyTorch", "Pandas", "NumPy", "Scikit-learn",
                "FastAPI", "Django", "Flask", "Spring Boot", "Node.js",
                "REST", "GraphQL", "Microservices", "API", "ETL",
                "Tableau", "Power BI", "Looker", "Excel", "R",
                "Agile", "Scrum", "DevOps", "CI/CD", "Testing"
            ]
            
            found_tech_skills = []
            job_text_lower = job_description.lower()
            for skill in tech_keywords:
                if skill.lower() in job_text_lower:
                    found_tech_skills.append(skill)
            
            entities["technologies"] = found_tech_skills
            nlp_insights["entities"] = entities
            
            # Sentiment analysis
            blob = TextBlob(job_description)
            nlp_insights["sentiment"] = {
                "polarity": blob.sentiment.polarity,  # -1 to 1
                "subjectivity": blob.sentiment.subjectivity,  # 0 to 1
                "tone": "positive" if blob.sentiment.polarity > 0.1 else "negative" if blob.sentiment.polarity < -0.1 else "neutral"
            }
            
            # Readability and complexity
            sentences = len(list(doc.sents))
            words = len([token for token in doc if not token.is_punct and not token.is_space])
            avg_sentence_length = words / sentences if sentences > 0 else 0
            
            nlp_insights["readability"] = {
                "sentence_count": sentences,
                "word_count": words,
                "avg_sentence_length": round(avg_sentence_length, 1),
                "complexity": "high" if avg_sentence_length > 20 else "medium" if avg_sentence_length > 15 else "low"
            }

        # Basic analysis (fallback/enhancement)
        word_count = len(job_description.split())
        char_count = len(job_description)
        
        # Urgency indicators
        urgency_keywords = ["urgent", "asap", "immediately", "start immediately", "urgent need"]
        urgency_score = sum(1 for keyword in urgency_keywords if keyword.lower() in job_description.lower())
        
        # Experience level detection
        if any(phrase in job_description.lower() for phrase in ["entry level", "junior", "0-1 year", "new grad"]):
            detected_level = "Entry Level"
        elif any(phrase in job_description.lower() for phrase in ["senior", "5+ years", "lead", "principal"]):
            detected_level = "Senior Level"
        else:
            detected_level = "Mid Level"

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Enhanced job description analysis complete!",
                "enhanced_analysis": {
                    "basic_metrics": {
                        "word_count": word_count,
                        "character_count": char_count,
                        "estimated_read_time": f"{max(1, word_count // 200)} min",
                        "urgency_score": urgency_score,
                        "detected_experience_level": detected_level
                    },
                    "gpt_insights": gpt_analysis,
                    "nlp_analysis": nlp_insights,
                    "ats_optimization": {
                        "keyword_density": len(nlp_insights.get("entities", {}).get("technologies", [])),
                        "readability_score": nlp_insights.get("readability", {}).get("complexity", "unknown"),
                        "sentiment_score": nlp_insights.get("sentiment", {}).get("tone", "neutral")
                    }
                },
                "job_description_preview": job_description[:300] + "..." if len(job_description) > 300 else job_description
            }
        )
        
    except Exception as e:
        print(f"Enhanced Analysis error: {str(e)}")
        # Fallback to basic analysis
        word_count = len(job_description.split())
        basic_skills = ["Python", "JavaScript", "SQL", "React", "AWS"]
        found_skills = [skill for skill in basic_skills if skill.lower() in job_description.lower()]
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Basic job description analysis (enhanced mode failed)",
                "enhanced_analysis": {
                    "basic_metrics": {
                        "word_count": word_count,
                        "character_count": len(job_description),
                        "found_skills": found_skills,
                        "detected_experience_level": "Mid Level"
                    },
                    "error": str(e)
                },
                "job_description_preview": job_description[:200] + "..." if len(job_description) > 200 else job_description
            }
        )

# NEW: Skills Gap Analysis Endpoint
@app.post("/api/skills-gap-analysis")
async def skills_gap_analysis(
    resume_text: str = Form(...),
    job_description: str = Form(...)
):
    """
    Analyze skills gap between resume and job requirements
    """
    try:
        gap_analysis_prompt = f"""
        As an expert career coach, analyze the gap between this resume and job requirements:

        RESUME:
        {resume_text[:2000]}  # Limit length for API

        JOB REQUIREMENTS:
        {job_description}

        Provide a JSON analysis:
        {{
            "skills_match": {{
                "matching_skills": ["skills present in both"],
                "missing_critical_skills": ["required skills not in resume"],
                "missing_preferred_skills": ["nice-to-have skills not in resume"],
                "additional_skills": ["skills in resume not mentioned in job"]
            }},
            "experience_gap": {{
                "experience_match": "excellent/good/partial/poor",
                "missing_experience": ["specific experience gaps"],
                "transferable_experience": ["relevant experience from different contexts"]
            }},
            "recommendations": {{
                "high_priority": ["most important improvements"],
                "medium_priority": ["helpful improvements"],
                "low_priority": ["nice-to-have improvements"]
            }},
            "match_score": {{
                "overall_match": "percentage like 75%",
                "skills_match": "percentage",
                "experience_match": "percentage"
            }}
        }}
        """

        gpt_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert career coach specializing in skills gap analysis and career development."},
                {"role": "user", "content": gap_analysis_prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )

        gap_analysis_raw = gpt_response.choices[0].message.content
        
        try:
            json_match = re.search(r'\{.*\}', gap_analysis_raw, re.DOTALL)
            if json_match:
                gap_analysis = json.loads(json_match.group())
            else:
                gap_analysis = {"error": "Could not parse response", "raw": gap_analysis_raw}
        except json.JSONDecodeError:
            gap_analysis = {"error": "Invalid JSON", "raw": gap_analysis_raw}

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Skills gap analysis complete!",
                "gap_analysis": gap_analysis,
                "analysis_timestamp": "2025-08-07"
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": "Skills gap analysis failed"
            }
        )

# NEW: AI Resume Customization Endpoint
@app.post("/api/customize-resume")
async def customize_resume(
    resume_text: str = Form(...),
    job_description: str = Form(...),
    customization_level: str = Form(default="medium")
):
    """
    Use GPT to customize resume based on job description
    """
    try:
        # Create customization prompt
        customization_prompt = f"""
        You are an expert resume writer and career coach. Customize this resume for the specific job description provided.

        ORIGINAL RESUME:
        {resume_text}

        TARGET JOB DESCRIPTION:
        {job_description}

        CUSTOMIZATION LEVEL: {customization_level}

        Please provide customized resume content with:
        1. Tailored professional summary
        2. Optimized experience bullet points (focus on relevant achievements)
        3. Prioritized skills section
        4. Keyword optimization for ATS systems
        5. Quantified achievements where possible

        Keep the same structure but optimize content for this specific role.
        Maintain truthfulness - only enhance and reframe existing content.

        Format your response as JSON with these keys:
        - customized_summary
        - optimized_experience
        - prioritized_skills
        - keyword_improvements
        - ats_score_improvement
        - customization_notes
        """

        # Replace your existing OpenAI call with:
        ai_result = await safe_openai_call(customization_prompt, 1500, 0.4)

        if ai_result["success"]:
            customized_content = ai_result["content"]
        else:
            # Fallback content generation
            customized_content = json.dumps({
                "customized_summary": "Professional with relevant experience for this role",
                "optimized_experience": ["Experience tailored for the target position"],
                "prioritized_skills": ["Key Skill 1", "Key Skill 2", "Key Skill 3"],
                "fallback_used": True,
                "fallback_reason": ai_result["user_message"]
            })

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Resume customized with AI!",
                "original_length": len(resume_text),
                "customization_level": customization_level,
                "customized_content": customized_content,
                "processing_time": "~3-5 seconds",
                "ai_confidence": "high"
            }
        )

    except Exception as e:
        print(f"GPT Customization error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error customizing resume: {str(e)}"
        )

# Test endpoint for GPT connection
@app.post("/api/test-gpt")
async def test_gpt():
    """Test GPT integration"""
    try:
        test_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'GPT integration successful!' if you can read this."}
            ],
            max_tokens=50
        )
        
        return {
            "success": True,
            "message": "GPT integration working!",
            "gpt_response": test_response.choices[0].message.content,
            "model": "gpt-3.5-turbo"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "GPT integration failed"
        }

# ADD THESE NEW ENDPOINTS TO YOUR main.py FILE
# Place these after your existing endpoints, before if __name__ == "__main__":

# NEW: Enhanced Template System Endpoints for Day 8

@app.post("/api/template-preview")
async def generate_template_preview(
    template_name: str = Form(...),
    color_scheme: str = Form(default="blue"),
    font_family: str = Form(default="Arial"),
    layout_style: str = Form(default="standard"),
    section_order: str = Form(default="standard"),
    resume_text: str = Form(default=""),
    job_description: str = Form(default="")
):
    """
    Generate template preview with customizations
    """
    try:
        # Template configurations
        template_configs = {
            "modern": {
                "name": "Modern Professional",
                "description": "Clean, ATS-friendly design perfect for tech and corporate roles",
                "features": ["ATS-Optimized", "Tech-Friendly", "Clean Design"],
                "best_for": ["Technology", "Consulting", "Corporate", "Startups"],
                "css_class": "resume-modern"
            },
            "classic": {
                "name": "Classic Executive", 
                "description": "Traditional format ideal for senior positions and conservative industries",
                "features": ["Executive-Level", "Conservative", "Traditional"],
                "best_for": ["Finance", "Law", "Government", "Senior Management"],
                "css_class": "resume-classic"
            },
            "creative": {
                "name": "Creative Designer",
                "description": "Eye-catching design perfect for creative roles",
                "features": ["Creative", "Visual Impact", "Modern"],
                "best_for": ["Design", "Marketing", "Media", "Creative Agencies"],
                "css_class": "resume-creative"
            },
            "minimal": {
                "name": "Minimal Clean",
                "description": "Ultra-clean design focusing on content clarity",
                "features": ["Minimalist", "Content-First", "Universal"], 
                "best_for": ["Any Industry", "Academic", "Research", "Consulting"],
                "css_class": "resume-minimal"
            },
            "executive": {
                "name": "Executive Premium",
                "description": "Sophisticated template for C-level executives",
                "features": ["C-Level", "Premium", "Leadership"],
                "best_for": ["Executive Roles", "Board Positions", "Senior Leadership"],
                "css_class": "resume-executive"
            }
        }
        
        # Color scheme mappings
        color_schemes = {
            "blue": {"primary": "#3498db", "secondary": "#2980b9", "accent": "#5dade2"},
            "green": {"primary": "#27ae60", "secondary": "#229954", "accent": "#58d68d"},
            "purple": {"primary": "#8e44ad", "secondary": "#7d3c98", "accent": "#af7ac5"},
            "orange": {"primary": "#e67e22", "secondary": "#d35400", "accent": "#f39c12"},
            "red": {"primary": "#e74c3c", "secondary": "#c0392b", "accent": "#ec7063"},
            "dark": {"primary": "#2c3e50", "secondary": "#1b2631", "accent": "#5d6d7e"}
        }
        
        # Generate template HTML
        template_html = generate_template_html(
            template_name, 
            template_configs[template_name],
            color_schemes[color_scheme],
            font_family,
            layout_style,
            section_order,
            resume_text,
            job_description
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Template preview generated successfully!",
                "template_info": template_configs[template_name],
                "customizations": {
                    "template": template_name,
                    "color_scheme": color_scheme,
                    "font_family": font_family,
                    "layout_style": layout_style,
                    "section_order": section_order
                },
                "template_html": template_html,
                "css_variables": {
                    "--primary-color": color_schemes[color_scheme]["primary"],
                    "--secondary-color": color_schemes[color_scheme]["secondary"],
                    "--accent-color": color_schemes[color_scheme]["accent"],
                    "--font-family": f"'{font_family}', sans-serif"
                }
            }
        )
        
    except Exception as e:
        print(f"Template preview error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating template preview: {str(e)}"
        )

def generate_template_html(template_name, template_config, colors, font, layout, section_order, resume_text="", job_description=""):
    """
    Generate HTML for specific template with customizations
    """
    
    # Extract or use sample data
    if resume_text:
        # In a real implementation, you'd parse the resume_text
        # For now, using sample data
        pass
    
    sample_data = {
        "name": "John Doe",
        "email": "john.doe@email.com", 
        "phone": "(555) 123-4567",
        "linkedin": "linkedin.com/in/johndoe",
        "summary": "Results-driven professional with 5+ years of experience in data analysis and strategic planning. Proven track record of delivering high-impact projects and driving organizational growth through innovative solutions.",
        "experience": [
            {
                "title": "Senior Business Analyst",
                "company": "Tech Solutions Inc.",
                "duration": "2022 - Present",
                "achievements": [
                    "Led data-driven initiatives that improved operational efficiency by 30%",
                    "Developed comprehensive analytical frameworks for strategic decision-making",
                    "Managed cross-functional teams of 8+ members to deliver complex projects",
                    "Collaborated with executive leadership to identify market opportunities"
                ]
            },
            {
                "title": "Business Analyst", 
                "company": "Innovation Consulting",
                "duration": "2020 - 2022",
                "achievements": [
                    "Conducted comprehensive market research and competitive analysis",
                    "Designed reporting systems that enhanced decision-making capabilities", 
                    "Optimized business processes resulting in 25% cost reduction"
                ]
            }
        ],
        "skills": ["Data Analysis", "Strategic Planning", "Project Management", "SQL", "Python", "Tableau", "Excel", "Leadership"],
        "education": {
            "degree": "Master of Business Analytics",
            "school": "University Name",
            "year": "2024",
            "details": "Magna Cum Laude | GPA: 3.8/4.0"
        },
        "certifications": [
            "Microsoft Power BI Data Analyst Associate",
            "Google Analytics Individual Qualification", 
            "AWS Certified Cloud Practitioner"
        ]
    }
    
    # Generate template-specific HTML
    if template_name == "modern":
        return generate_modern_template(sample_data, colors, font)
    elif template_name == "minimal":
        return generate_minimal_template(sample_data, colors, font)
    elif template_name == "executive":
        return generate_executive_template(sample_data, colors, font)
    else:
        return generate_modern_template(sample_data, colors, font)

def generate_modern_template(data, colors, font):
    """Generate Modern Professional template HTML"""
    return f'''
    <div class="resume-modern" style="font-family: '{font}', sans-serif; padding: 40px; background: white; color: #333;">
        <div class="header-section" style="text-align: center; border-bottom: 3px solid {colors['primary']}; padding-bottom: 20px; margin-bottom: 30px;">
            <h1 style="font-size: 2.5rem; color: #2c3e50; margin-bottom: 10px; font-weight: bold;">{data['name']}</h1>
            <p style="color: #7f8c8d; font-size: 1.1rem;">{data['email']} | {data['phone']} | {data['linkedin']}</p>
        </div>
        
        <div class="section" style="margin-bottom: 25px; padding: 20px; background: rgba({colors['primary'][1:]}, 0.1); border-radius: 10px; border-left: 4px solid {colors['primary']};">
            <h2 style="color: {colors['primary']}; font-size: 1.3rem; margin-bottom: 15px; font-weight: bold;">Professional Summary</h2>
            <p style="line-height: 1.6; margin: 0;">{data['summary']}</p>
        </div>
        
        <div class="section" style="margin-bottom: 25px; padding: 20px; background: rgba({colors['primary'][1:]}, 0.1); border-radius: 10px; border-left: 4px solid {colors['primary']};">
            <h2 style="color: {colors['primary']}; font-size: 1.3rem; margin-bottom: 15px; font-weight: bold;">Professional Experience</h2>
            {''.join([f'''
            <div style="margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <h3 style="color: #2c3e50; font-size: 1.1rem; margin: 0; font-weight: 600;">{exp['title']}</h3>
                    <span style="color: #7f8c8d; font-style: italic;">{exp['duration']}</span>
                </div>
                <p style="color: {colors['secondary']}; margin: 5px 0 10px 0; font-weight: 500;">{exp['company']}</p>
                <ul style="margin: 0; padding-left: 20px; color: #555;">
                    {''.join([f'<li style="margin-bottom: 5px;">{achievement}</li>' for achievement in exp['achievements']])}
                </ul>
            </div>
            ''' for exp in data['experience']])}
        </div>
        
        <div class="section" style="margin-bottom: 25px; padding: 20px; background: rgba({colors['primary'][1:]}, 0.1); border-radius: 10px; border-left: 4px solid {colors['primary']};">
            <h2 style="color: {colors['primary']}; font-size: 1.3rem; margin-bottom: 15px; font-weight: bold;">Core Skills</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                {''.join([f'<div style="background: linear-gradient(135deg, {colors['secondary']}, {colors['primary']}); color: white; padding: 8px 16px; border-radius: 20px; text-align: center; font-weight: 500;">{skill}</div>' for skill in data['skills']])}
            </div>
        </div>
        
        <div class="section" style="margin-bottom: 25px; padding: 20px; background: rgba({colors['primary'][1:]}, 0.1); border-radius: 10px; border-left: 4px solid {colors['primary']};">
            <h2 style="color: {colors['primary']}; font-size: 1.3rem; margin-bottom: 15px; font-weight: bold;">Education</h2>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="color: #2c3e50; margin: 0; font-weight: 600;">{data['education']['degree']}</h3>
                    <p style="color: {colors['secondary']}; margin: 5px 0; font-weight: 500;">{data['education']['school']}</p>
                    <p style="color: #666; margin: 0; font-style: italic;">{data['education']['details']}</p>
                </div>
                <span style="color: #7f8c8d; font-weight: 500;">{data['education']['year']}</span>
            </div>
        </div>
        
        <div class="section" style="padding: 20px; background: rgba({colors['primary'][1:]}, 0.1); border-radius: 10px; border-left: 4px solid {colors['primary']};">
            <h2 style="color: {colors['primary']}; font-size: 1.3rem; margin-bottom: 15px; font-weight: bold;">Certifications</h2>
            <ul style="margin: 0; padding-left: 20px; color: #555;">
                {''.join([f'<li style="margin-bottom: 5px;">{cert}</li>' for cert in data['certifications']])}
            </ul>
        </div>
    </div>
    '''

def generate_classic_template(data, colors, font):
    """Generate Classic Executive template HTML"""
    return f'''
    <div class="resume-classic" style="font-family: '{font}', serif; padding: 40px; background: white; color: #333; line-height: 1.7;">
        <div class="header-section" style="text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px;">
            <h1 style="font-size: 2.2rem; color: #333; margin-bottom: 10px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;">{data['name']}</h1>
            <p style="color: #666; font-size: 1rem;">{data['email']} | {data['phone']} | {data['linkedin']}</p>
        </div>
        
        <div style="margin-bottom: 25px;">
            <h2 style="color: #333; border-bottom: 1px solid #ccc; padding-bottom: 5px; text-transform: uppercase; font-size: 1.1rem; letter-spacing: 0.5px; margin-bottom: 15px;">Professional Summary</h2>
            <p style="text-align: justify; margin: 0;">{data['summary']}</p>
        </div>
        
        <div style="margin-bottom: 25px;">
            <h2 style="color: #333; border-bottom: 1px solid #ccc; padding-bottom: 5px; text-transform: uppercase; font-size: 1.1rem; letter-spacing: 0.5px; margin-bottom: 15px;">Professional Experience</h2>
            {''.join([f'''
            <div style="margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <h3 style="color: #333; font-size: 1rem; margin: 0; font-weight: bold; text-transform: uppercase;">{exp['title']}</h3>
                    <span style="color: #666; font-style: italic;">{exp['duration']}</span>
                </div>
                <p style="color: #666; margin: 5px 0; font-style: italic;">{exp['company']}</p>
                <ul style="margin: 10px 0 0 20px;">
                    {''.join([f'<li style="margin-bottom: 5px;">{achievement}</li>' for achievement in exp['achievements']])}
                </ul>
            </div>
            ''' for exp in data['experience']])}
        </div>
        
        <div style="margin-bottom: 25px;">
            <h2 style="color: #333; border-bottom: 1px solid #ccc; padding-bottom: 5px; text-transform: uppercase; font-size: 1.1rem; letter-spacing: 0.5px; margin-bottom: 15px;">Core Competencies</h2>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px;">
                {''.join([f'<div style="padding: 5px 0;">• {skill}</div>' for skill in data['skills']])}
            </div>
        </div>
        
        <div style="margin-bottom: 25px;">
            <h2 style="color: #333; border-bottom: 1px solid #ccc; padding-bottom: 5px; text-transform: uppercase; font-size: 1.1rem; letter-spacing: 0.5px; margin-bottom: 15px;">Education</h2>
            <div>
                <h3 style="color: #333; margin: 0; font-weight: bold; text-transform: uppercase;">{data['education']['degree']}</h3>
                <p style="color: #666; margin: 5px 0;">{data['education']['school']}, {data['education']['year']}</p>
                <p style="color: #666; margin: 0; font-style: italic;">{data['education']['details']}</p>
            </div>
        </div>
        
        <div>
            <h2 style="color: #333; border-bottom: 1px solid #ccc; padding-bottom: 5px; text-transform: uppercase; font-size: 1.1rem; letter-spacing: 0.5px; margin-bottom: 15px;">Certifications</h2>
            <div>
                {''.join([f'<p style="margin: 5px 0;">• {cert}</p>' for cert in data['certifications']])}
            </div>
        </div>
    </div>
    '''

def generate_creative_template(data, colors, font):
    """Generate Creative Designer template HTML"""
    return f'''
    <div class="resume-creative" style="font-family: '{font}', sans-serif; background: linear-gradient(135deg, {colors['primary']}, {colors['secondary']}); color: white; border-radius: 15px; overflow: hidden;">
        <div class="header-section" style="padding: 30px; text-align: center; position: relative;">
            <h1 style="font-size: 2.5rem; margin-bottom: 10px; font-weight: bold; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">{data['name']}</h1>
            <p style="font-size: 1.2rem; opacity: 0.9; margin-bottom: 5px;">Creative Professional & Strategic Innovator</p>
            <p style="opacity: 0.8;">{data['email']} | {data['phone']} | Portfolio: johndoe.design</p>
        </div>
        
        <div style="padding: 30px; background: rgba(255,255,255,0.1); margin: 0 20px 20px 20px; border-radius: 15px; backdrop-filter: blur(10px);">
            <h2 style="font-size: 1.4rem; margin-bottom: 15px; font-weight: bold;">💫 Creative Vision</h2>
            <p style="line-height: 1.7; margin: 0;">{data['summary']}</p>
        </div>
        
        <div style="padding: 30px; background: rgba(255,255,255,0.1); margin: 0 20px 20px 20px; border-radius: 15px; backdrop-filter: blur(10px);">
            <h2 style="font-size: 1.4rem; margin-bottom: 15px; font-weight: bold;">🚀 Creative Journey</h2>
            {''.join([f'''
            <div style="border-left: 4px solid rgba(255,255,255,0.5); padding-left: 20px; margin-bottom: 20px; position: relative;">
                <div style="position: absolute; left: -8px; top: 10px; width: 12px; height: 12px; background: white; border-radius: 50%;"></div>
                <h3 style="margin: 0; font-size: 1.1rem;">{exp['title']}</h3>
                <p style="margin: 5px 0; opacity: 0.9; font-weight: 600;">{exp['company']} • {exp['duration']}</p>
                <ul style="margin: 10px 0 0 20px; opacity: 0.9;">
                    {''.join([f'<li style="margin-bottom: 5px;">{achievement}</li>' for achievement in exp['achievements']])}
                </ul>
            </div>
            ''' for exp in data['experience']])}
        </div>
        
        <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 20px; padding: 0 20px 20px 20px;">
            <div style="background: rgba(255,255,255,0.1); padding: 25px; border-radius: 15px; backdrop-filter: blur(10px);">
                <h2 style="font-size: 1.4rem; margin-bottom: 15px; font-weight: bold;">⚡ Creative Skills</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
                    {''.join([f'<div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 20px; text-align: center; font-weight: 500;">{skill}</div>' for skill in data['skills']])}
                </div>
            </div>
            
            <div>
                <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; margin-bottom: 15px; backdrop-filter: blur(10px);">
                    <h3 style="font-size: 1.2rem; margin-bottom: 10px; text-align: center;">🎓 Education</h3>
                    <div style="text-align: center;">
                        <p style="font-weight: bold; margin: 0;">{data['education']['degree']}</p>
                        <p style="margin: 5px 0; opacity: 0.9;">{data['education']['school']}</p>
                        <p style="margin: 0; opacity: 0.8;">{data['education']['year']}</p>
                    </div>
                </div>
                
                <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; backdrop-filter: blur(10px);">
                    <h3 style="font-size: 1.2rem; margin-bottom: 10px; text-align: center;">🏆 Achievements</h3>
                    <div style="font-size: 0.9rem; opacity: 0.9;">
                        {''.join([f'<p style="margin: 8px 0;">🎨 {cert.split()[0]} Award</p>' for cert in data['certifications'][:3]])}
                    </div>
                </div>
            </div>
        </div>
    </div>
    '''

def generate_minimal_template(data, colors, font):
    """Generate Minimal Clean template HTML"""
    return f'''
    <div class="resume-minimal" style="font-family: '{font}', sans-serif; padding: 40px; background: white; color: #333; font-weight: 300;">
        <div style="border-bottom: 1px solid #eee; padding-bottom: 30px; margin-bottom: 30px;">
            <h1 style="font-size: 2.2rem; color: #333; margin: 0; font-weight: 300; letter-spacing: -1px;">{data['name']}</h1>
            <p style="color: #777; font-size: 1rem; margin: 8px 0 0 0; font-weight: 300;">{data['email']} • {data['phone']} • {data['linkedin']}</p>
        </div>
        
        <div style="margin-bottom: 40px;">
            <h2 style="font-size: 1.1rem; color: #333; font-weight: 600; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px;">Summary</h2>
            <p style="line-height: 1.8; color: #555; margin: 0;">{data['summary']}</p>
        </div>
        
        <div style="margin-bottom: 40px;">
            <h2 style="font-size: 1.1rem; color: #333; font-weight: 600; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 1px;">Experience</h2>
            {''.join([f'''
            <div style="margin-bottom: 25px;">
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 5px;">
                    <h3 style="margin: 0; font-size: 1rem; color: #333; font-weight: 500;">{exp['title']}</h3>
                    <span style="color: #777; font-size: 0.9rem; font-weight: 300;">{exp['duration']}</span>
                </div>
                <p style="margin: 0 0 10px 0; color: #777; font-size: 0.95rem; font-weight: 300;">{exp['company']}</p>
                <ul style="margin: 0; padding-left: 20px; color: #555; line-height: 1.7;">
                    {''.join([f'<li style="margin-bottom: 8px;">{achievement}</li>' for achievement in exp['achievements']])}
                </ul>
            </div>
            ''' for exp in data['experience']])}
        </div>
        
        <div style="margin-bottom: 40px;">
            <h2 style="font-size: 1.1rem; color: #333; font-weight: 600; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px;">Skills</h2>
            <div style="color: #555; font-weight: 300;">
                {' • '.join(data['skills'])}
            </div>
        </div>
        
        <div style="margin-bottom: 40px;">
            <h2 style="font-size: 1.1rem; color: #333; font-weight: 600; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px;">Education</h2>
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <div>
                    <p style="margin: 0; font-weight: 500; color: #333;">{data['education']['degree']}</p>
                    <p style="margin: 5px 0 0 0; color: #777; font-weight: 300;">{data['education']['school']}</p>
                </div>
                <span style="color: #777; font-size: 0.9rem; font-weight: 300;">{data['education']['year']}</span>
            </div>
        </div>
        
        <div>
            <h2 style="font-size: 1.1rem; color: #333; font-weight: 600; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px;">Certifications</h2>
            <div style="color: #555; font-weight: 300; line-height: 1.6;">
                {''.join([f'<p style="margin: 5px 0;">{cert}</p>' for cert in data['certifications']])}
            </div>
        </div>
    </div>
    '''

def generate_executive_template(data, colors, font):
    """Generate Executive Premium template HTML"""
    return f'''
    <div class="resume-executive" style="font-family: '{font}', serif; background: linear-gradient(to bottom, #f8f9fa 0%, #ffffff 100%); border: 1px solid #dee2e6; overflow: hidden;">
        <div style="background: {colors['primary']}; color: white; padding: 30px; text-align: center;">
            <h1 style="font-size: 2.2rem; margin-bottom: 8px; font-weight: bold; letter-spacing: 1px;">{data['name']}</h1>
            <p style="font-size: 1.1rem; opacity: 0.9; margin-bottom: 5px;">Chief Executive Officer</p>
            <p style="opacity: 0.8;">{data['email']} | {data['phone']} | {data['linkedin']}</p>
        </div>
        
        <div style="padding: 40px;">
            <div style="margin-bottom: 30px;">
                <h2 style="color: {colors['primary']}; font-size: 1.3rem; font-weight: bold; border-bottom: 2px solid {colors['primary']}; padding-bottom: 8px; margin-bottom: 20px;">Executive Summary</h2>
                <p style="line-height: 1.7; color: #2c3e50; text-align: justify; margin: 0;">{data['summary']}</p>
            </div>
            
            <div style="margin-bottom: 30px;">
                <h2 style="color: {colors['primary']}; font-size: 1.3rem; font-weight: bold; border-bottom: 2px solid {colors['primary']}; padding-bottom: 8px; margin-bottom: 20px;">Leadership Experience</h2>
                {''.join([f'''
                <div style="margin-bottom: 25px; padding: 20px; background: rgba({colors['primary'][1:]}, 0.05); border-radius: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h3 style="color: {colors['primary']}; font-size: 1.2rem; margin: 0; font-weight: bold;">{exp['title']}</h3>
                        <span style="color: #666; font-weight: 600; font-style: italic;">{exp['duration']}</span>
                    </div>
                    <p style="color: {colors['secondary']}; margin: 5px 0 15px 0; font-weight: 600; font-size: 1.1rem;">{exp['company']}</p>
                    <ul style="margin: 0; padding-left: 25px; color: #444; line-height: 1.8;">
                        {''.join([f'<li style="margin-bottom: 8px; font-weight: 400;">{achievement}</li>' for achievement in exp['achievements']])}
                    </ul>
                </div>
                ''' for exp in data['experience']])}
            </div>
            
            <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 30px; margin-bottom: 30px;">
                <div>
                    <h2 style="color: {colors['primary']}; font-size: 1.3rem; font-weight: bold; border-bottom: 2px solid {colors['primary']}; padding-bottom: 8px; margin-bottom: 20px;">Core Competencies</h2>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
                        {''.join([f'<div style="background: linear-gradient(135deg, {colors['primary']}, {colors['secondary']}); color: white; padding: 12px 16px; border-radius: 8px; text-align: center; font-weight: 600; font-size: 0.95rem;">{skill}</div>' for skill in data['skills']])}
                    </div>
                </div>
                
                <div>
                    <h2 style="color: {colors['primary']}; font-size: 1.3rem; font-weight: bold; border-bottom: 2px solid {colors['primary']}; padding-bottom: 8px; margin-bottom: 20px;">Education</h2>
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid {colors['primary']};">
                        <h3 style="color: {colors['primary']}; margin: 0 0 8px 0; font-weight: bold;">{data['education']['degree']}</h3>
                        <p style="color: #666; margin: 5px 0; font-weight: 600;">{data['education']['school']}</p>
                        <p style="color: #888; margin: 5px 0 0 0; font-style: italic;">{data['education']['details']}</p>
                        <p style="color: {colors['secondary']}; margin: 10px 0 0 0; font-weight: bold;">{data['education']['year']}</p>
                    </div>
                </div>
            </div>
            
            <div>
                <h2 style="color: {colors['primary']}; font-size: 1.3rem; font-weight: bold; border-bottom: 2px solid {colors['primary']}; padding-bottom: 8px; margin-bottom: 20px;">Professional Certifications</h2>
                <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid {colors['primary']};">
                    {''.join([f'<div style="margin-bottom: 10px; padding: 8px 0; border-bottom: 1px solid #eee; font-weight: 500; color: #555;">🏆 {cert}</div>' for cert in data['certifications']])}
                </div>
            </div>
        </div>
    </div>
    '''

# NEW: Template Selection and Management Endpoint
@app.post("/api/select-template")
async def select_template(
    template_name: str = Form(...),
    user_preferences: str = Form(default="{}"),
    save_selection: bool = Form(default=True)
):
    """
    Handle template selection and save user preferences
    """
    try:
        template_info = {
            "modern": {
                "name": "Modern Professional",
                "category": "Professional",
                "industry_fit": ["Technology", "Consulting", "Corporate"],
                "ats_score": 95,
                "complexity": "Medium"
            },
            "classic": {
                "name": "Classic Executive",
                "category": "Traditional", 
                "industry_fit": ["Finance", "Law", "Government"],
                "ats_score": 98,
                "complexity": "Low"
            },
            "creative": {
                "name": "Creative Designer",
                "category": "Creative",
                "industry_fit": ["Design", "Marketing", "Media"],
                "ats_score": 85,
                "complexity": "High"
            },
            "minimal": {
                "name": "Minimal Clean",
                "category": "Universal",
                "industry_fit": ["Any Industry", "Academic", "Research"],
                "ats_score": 92,
                "complexity": "Low"
            },
            "executive": {
                "name": "Executive Premium", 
                "category": "Executive",
                "industry_fit": ["C-Level", "Senior Management", "Board"],
                "ats_score": 90,
                "complexity": "Medium"
            }
        }
        
        if template_name not in template_info:
            raise HTTPException(status_code=400, detail="Invalid template name")
        
        selected_template = template_info[template_name]
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"{selected_template['name']} template selected successfully!",
                "template_details": {
                    "name": template_name,
                    "info": selected_template,
                    "customization_options": {
                        "colors": ["blue", "green", "purple", "orange", "red", "dark"],
                        "fonts": ["Arial", "Times New Roman", "Calibri", "Georgia", "Helvetica"],
                        "layouts": ["standard", "sidebar", "two-column", "header-focus"],
                        "section_orders": ["standard", "skills-first", "experience-first", "education-first"]
                    }
                },
                "next_steps": [
                    "Customize colors and fonts",
                    "Generate resume preview",
                    "Export to PDF/DOCX"
                ]
            }
        )
        
    except Exception as e:
        print(f"Template selection error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error selecting template: {str(e)}"
        )

# NEW: Template Analytics and Recommendations
@app.post("/api/template-recommendations")
async def get_template_recommendations(
    job_description: str = Form(...),
    industry: str = Form(default=""),
    experience_level: str = Form(default="mid")
):
    """
    Get AI-powered template recommendations based on job description and user profile
    """
    try:
        # Analyze job description for industry and role type
        analysis_prompt = f"""
        Analyze this job description and recommend the best resume template:

        Job Description: {job_description}
        Industry: {industry}
        Experience Level: {experience_level}

        Available templates:
        1. Modern Professional - ATS-friendly, tech/corporate
        2. Classic Executive - Traditional, conservative industries  
        3. Creative Designer - Visual, creative roles
        4. Minimal Clean - Universal, content-focused
        5. Executive Premium - C-level, senior leadership

        Provide recommendations as JSON:
        {{
            "primary_recommendation": "template_name",
            "confidence": "percentage",
            "reasoning": "explanation",
            "alternative_options": ["template2", "template3"],
            "industry_analysis": "detected industry/role type",
            "template_scores": {{
                "modern": score,
                "classic": score,
                "creative": score,
                "minimal": score,
                "executive": score
            }}
        }}
        """

        gpt_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert resume template advisor with deep knowledge of industry standards and ATS requirements."},
                {"role": "user", "content": analysis_prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )

        recommendations_raw = gpt_response.choices[0].message.content
        
        try:
            json_match = re.search(r'\{.*\}', recommendations_raw, re.DOTALL)
            if json_match:
                recommendations = json.loads(json_match.group())
            else:
                recommendations = {"error": "Could not parse recommendations", "raw": recommendations_raw}
        except json.JSONDecodeError:
            recommendations = {"error": "Invalid JSON", "raw": recommendations_raw}

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Template recommendations generated successfully!",
                "recommendations": recommendations,
                "metadata": {
                    "job_analysis": f"Analyzed {len(job_description.split())} words",
                    "industry_context": industry or "Auto-detected",
                    "experience_level": experience_level,
                    "recommendation_confidence": recommendations.get("confidence", "85%")
                }
            }
        )

    except Exception as e:
        print(f"Template recommendations error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )

# NEW: Batch Template Generation for A/B Testing
@app.post("/api/generate-multiple-templates")
async def generate_multiple_templates(
    resume_text: str = Form(...),
    job_description: str = Form(...),
    template_list: str = Form(default="modern,classic,minimal")
):
    """
    Generate multiple template versions for comparison
    """
    try:
        templates = template_list.split(",")
        results = {}
        
        for template_name in templates:
            if template_name.strip() in ["modern", "classic", "creative", "minimal", "executive"]:
                # Generate each template
                template_html = generate_template_html(
                    template_name.strip(),
                    {"name": template_name.strip(), "features": []},
                    {"primary": "#3498db", "secondary": "#2980b9", "accent": "#5dade2"},
                    "Arial",
                    "standard", 
                    "standard",
                    resume_text,
                    job_description
                )
                
                results[template_name.strip()] = {
                    "html": template_html,
                    "generated_at": "2025-08-13",
                    "ats_score": 90 + len(template_name),  # Simulated score
                    "readability": "High"
                }
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Generated {len(results)} template variations successfully!",
                "templates": results,
                "comparison_metrics": {
                    "total_generated": len(results),
                    "processing_time": "3-5 seconds",
                    "recommended_for_testing": "Use A/B testing to see which performs best"
                }
            }
        )

    except Exception as e:
        print(f"Batch template generation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating multiple templates: {str(e)}"
        )

# Enhanced customization endpoint
@app.post("/api/apply-template-customizations")
async def apply_template_customizations(
    template_name: str = Form(...),
    customizations: str = Form(...)  # JSON string with all customizations
):
    """
    Apply comprehensive customizations to selected template
    """
    try:
        custom_config = json.loads(customizations)
        
        # Validate customizations
        valid_colors = ["blue", "green", "purple", "orange", "red", "dark"]
        valid_fonts = ["Arial", "Times New Roman", "Calibri", "Georgia", "Helvetica"]
        valid_layouts = ["standard", "sidebar", "two-column", "header-focus"]
        
        color = custom_config.get("color", "blue")
        font = custom_config.get("font", "Arial") 
        layout = custom_config.get("layout", "standard")
        
        if color not in valid_colors:
            color = "blue"
        if font not in valid_fonts:
            font = "Arial"
        if layout not in valid_layouts:
            layout = "standard"
            
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Template customizations applied successfully!",
                "applied_customizations": {
                    "template": template_name,
                    "color_scheme": color,
                    "font_family": font,
                    "layout_style": layout,
                    "timestamp": "2025-08-13"
                },
                "css_updates": {
                    "--primary-color": {"blue": "#3498db", "green": "#27ae60", "purple": "#8e44ad", "orange": "#e67e22", "red": "#e74c3c", "dark": "#2c3e50"}[color],
                    "--font-family": f"'{font}', sans-serif"
                },
                "ready_for_export": True
            }
        )
        
    except Exception as e:
        print(f"Template customization error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error applying customizations: {str(e)}"
        )

# NEW: Cover Letter Generation Endpoint
@app.post("/api/generate-cover-letter")
async def generate_cover_letter(
    resume_text: str = Form(...),
    job_description: str = Form(...),
    company_name: str = Form(...),
    position_title: str = Form(...),
    tone: str = Form(default="professional")  # professional, enthusiastic, confident
):
    """
    Generate AI-powered cover letter based on resume and job description
    """
    try:
        cover_letter_prompt = f"""
        You are an expert career coach and professional writer. Create a compelling cover letter based on the provided information.

        CANDIDATE'S RESUME:
        {resume_text[:1500]}  # Limit for API efficiency

        TARGET JOB:
        Company: {company_name}
        Position: {position_title}
        Job Description: {job_description}

        TONE: {tone}

        Please create a professional cover letter with:
        1. Engaging opening paragraph that shows enthusiasm for the specific role
        2. 2-3 middle paragraphs highlighting relevant experience and skills
        3. Strong closing paragraph with call to action
        4. Proper business letter formatting
        5. Personalized content referencing the company and role

        Keep it concise (3-4 paragraphs), professional, and compelling.
        Make sure it complements the resume without repeating everything.

        Format as JSON with these keys:
        - opening_paragraph
        - body_paragraphs (array)
        - closing_paragraph
        - suggested_subject_line
        - key_highlights (array of 3-4 main selling points)
        """

        gpt_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert career coach with 15+ years of experience writing compelling cover letters that get interviews."},
                {"role": "user", "content": cover_letter_prompt}
            ],
            max_tokens=1000,
            temperature=0.4
        )

        cover_letter_content = gpt_response.choices[0].message.content

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Cover letter generated successfully!",
                "company_name": company_name,
                "position_title": position_title,
                "tone": tone,
                "cover_letter_content": cover_letter_content,
                "estimated_reading_time": "45-60 seconds",
                "ai_confidence": "high"
            }
        )

    except Exception as e:
        print(f"Cover letter generation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating cover letter: {str(e)}"
        )

# NEW: Resume Scoring and Analytics Endpoint
@app.post("/api/analyze-resume-score")
async def analyze_resume_score(
    resume_text: str = Form(...),
    job_description: str = Form(...),
    target_keywords: str = Form(default="")
):
    """
    Provide detailed resume scoring and improvement analytics
    """
    try:
        scoring_prompt = f"""
        As an expert ATS system analyst and recruiting specialist, provide a comprehensive score for this resume against the job requirements.

        RESUME CONTENT:
        {resume_text[:2000]}

        JOB REQUIREMENTS:
        {job_description}

        TARGET KEYWORDS: {target_keywords}

        Provide detailed scoring analysis as JSON:
        {{
            "overall_score": "percentage out of 100",
            "category_scores": {{
                "keyword_match": "percentage",
                "experience_relevance": "percentage", 
                "skills_alignment": "percentage",
                "format_optimization": "percentage",
                "ats_compatibility": "percentage"
            }},
            "strengths": ["list of 3-4 strong points"],
            "improvement_areas": ["list of 3-4 specific improvements needed"],
            "missing_keywords": ["important keywords not found in resume"],
            "keyword_frequency": {{
                "high_value_keywords": ["keywords that appear multiple times"],
                "underused_keywords": ["important keywords that should appear more"]
            }},
            "ats_recommendations": ["specific formatting and content suggestions"],
            "competitive_analysis": {{
                "estimated_ranking": "how this resume would rank against other candidates",
                "interview_likelihood": "low/medium/high based on current state"
            }}
        }}
        """

        gpt_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert ATS analyst with deep knowledge of recruiting systems and candidate evaluation."},
                {"role": "user", "content": scoring_prompt}
            ],
            max_tokens=1200,
            temperature=0.2
        )

        scoring_analysis = gpt_response.choices[0].message.content

        # Additional basic analysis
        resume_word_count = len(resume_text.split())
        job_word_count = len(job_description.split())
        
        # Simple keyword matching for backup scoring
        job_words = set(job_description.lower().split())
        resume_words = set(resume_text.lower().split())
        common_words = job_words.intersection(resume_words)
        basic_match_score = min(100, int((len(common_words) / len(job_words)) * 100))

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Resume scoring analysis complete!",
                "detailed_analysis": scoring_analysis,
                "basic_metrics": {
                    "resume_word_count": resume_word_count,
                    "job_description_word_count": job_word_count,
                    "basic_keyword_match": f"{basic_match_score}%",
                    "analysis_timestamp": "2025-08-08"
                },
                "improvement_priority": "high" if basic_match_score < 60 else "medium" if basic_match_score < 80 else "low"
            }
        )

    except Exception as e:
        print(f"Resume scoring error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing resume score: {str(e)}"
        )

# NEW: LinkedIn Profile Integration Endpoint
@app.post("/api/linkedin-integration")
async def linkedin_integration(
    linkedin_url: str = Form(...),
    extraction_type: str = Form(default="basic")  # basic, detailed, skills_only
):
    """
    Simulate LinkedIn profile data extraction (placeholder for actual LinkedIn API)
    """
    try:
        # This is a simulation - real LinkedIn integration would require OAuth and API keys
        simulated_linkedin_data = {
            "profile_url": linkedin_url,
            "extraction_type": extraction_type,
            "extracted_data": {
                "headline": "Business Analytics Professional | Data-Driven Decision Making",
                "summary": "Results-driven professional with expertise in data analysis and business intelligence",
                "experience": [
                    {
                        "title": "Business Analyst",
                        "company": "Previous Company",
                        "duration": "2022 - Present",
                        "description": "Led data analysis initiatives and business process optimization"
                    }
                ],
                "skills": [
                    "Data Analysis", "Business Intelligence", "SQL", "Python", 
                    "Tableau", "Power BI", "Excel", "Project Management"
                ],
                "education": [
                    {
                        "degree": "Master of Business Analytics",
                        "institution": "University Name",
                        "year": "2024"
                    }
                ],
                "certifications": [
                    "Microsoft Power BI Certification",
                    "Google Analytics Certified"
                ]
            },
            "integration_suggestions": [
                "Add LinkedIn headline to resume summary",
                "Include recent certifications in skills section",
                "Highlight quantified achievements from experience",
                "Optimize skills section based on LinkedIn endorsements"
            ]
        }

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "LinkedIn profile data extracted successfully!",
                "profile_data": simulated_linkedin_data,
                "integration_options": {
                    "auto_populate": "Automatically fill resume sections",
                    "merge_skills": "Combine LinkedIn skills with resume",
                    "sync_experience": "Update experience section",
                    "import_summary": "Import LinkedIn summary"
                },
                "note": "This is a simulation - real LinkedIn integration requires API access"
            }
        )

    except Exception as e:
        print(f"LinkedIn integration error: {str(e)}")
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "message": "LinkedIn integration simulation",
                "error": str(e),
                "note": "This endpoint simulates LinkedIn integration functionality"
            }
        )

# NEW: User Feedback and Rating Endpoint
@app.post("/api/submit-feedback")
async def submit_feedback(
    rating: int = Form(...),  # 1-5 stars
    feedback_text: str = Form(...),
    feature_used: str = Form(...),  # resume_customization, template_selection, pdf_export, etc.
    improvement_suggestions: str = Form(default=""),
    user_email: str = Form(default="anonymous")
):
    """
    Collect user feedback and ratings for product improvement
    """
    try:
        feedback_data = {
            "rating": rating,
            "feedback_text": feedback_text,
            "feature_used": feature_used,
            "improvement_suggestions": improvement_suggestions,
            "user_email": user_email if user_email != "anonymous" else "anonymous",
            "timestamp": "2025-08-08",
            "session_id": "generated_session_id"
        }

        # In a real application, this would save to a database
        # For now, we'll just log it and return confirmation
        print(f"User Feedback Received: {feedback_data}")

        # Generate response based on feedback
        if rating >= 4:
            response_message = "Thank you for the positive feedback! We're thrilled you're loving the tool."
        elif rating == 3:
            response_message = "Thanks for the feedback! We're always working to improve your experience."
        else:
            response_message = "Thank you for the honest feedback. We'll use this to make the tool better!"

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Feedback submitted successfully!",
                "response": response_message,
                "feedback_id": "fb_" + str(hash(feedback_text))[:8],
                "follow_up": {
                    "will_contact": rating <= 2,  # Follow up on poor ratings
                    "estimated_improvements": "Next update in 1-2 weeks",
                    "beta_testing": "Would you like to test new features early?"
                }
            }
        )

    except Exception as e:
        print(f"Feedback submission error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error submitting feedback: {str(e)}"
        )

# NEW: Advanced Template Customization Endpoint
@app.post("/api/customize-template")
async def customize_template(
    template_name: str = Form(...),
    color_scheme: str = Form(default="default"),  # default, blue, green, purple, dark
    font_family: str = Form(default="Arial"),     # Arial, Times, Calibri, Modern
    layout_style: str = Form(default="standard"), # standard, sidebar, creative
    section_order: str = Form(default="standard") # standard, skills_first, experience_first
):
    """
    Generate customized template configurations
    """
    try:
        customization_config = {
            "template_name": template_name,
            "customizations": {
                "color_scheme": color_scheme,
                "font_family": font_family,
                "layout_style": layout_style,
                "section_order": section_order
            },
            "css_modifications": {
                "primary_color": "#3498db" if color_scheme == "blue" else "#27ae60" if color_scheme == "green" else "#8e44ad" if color_scheme == "purple" else "#2c3e50",
                "font_stack": f"'{font_family}', sans-serif",
                "layout_grid": "sidebar" if layout_style == "sidebar" else "traditional"
            },
            "preview_ready": True
        }

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Template customization ready!",
                "customization": customization_config,
                "estimated_generation_time": "2-3 seconds",
                "preview_available": True
            }
        )

    except Exception as e:
        print(f"Template customization error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error customizing template: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    PORT = int(os.getenv("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=PORT)