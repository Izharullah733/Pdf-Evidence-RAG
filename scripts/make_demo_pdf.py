"""Generate the controlled, three-page source used by the demo and evaluation."""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

PAGES = [
    (
        "Northstar Learning Lab - Course Guide",
        [
            "The Applied AI course lasts eight weeks.",
            "Classes meet every Tuesday and Thursday at 18:00.",
            "The course coordinator is Dr. Sara Malik.",
            "The support desk opens at 09:00 and closes at 17:00 on weekdays.",
            "This guide applies to the autumn training cohort.",
        ],
    ),
    (
        "Assessment and Submission",
        [
            "The final project contributes 40 percent of the course grade.",
            "Weekly exercises contribute 35 percent and the final quiz contributes 25 percent.",
            "Students must submit a technical report and a demonstration video.",
            "The technical report must be between three and five pages.",
            "The demonstration video must be between five and seven minutes.",
            "Late submissions are accepted for two days with a 10 percent penalty.",
        ],
    ),
    (
        "Lab Access and Project Rules",
        [
            "The computer lab is open from 08:00 to 20:00 on weekdays.",
            "Students must book a lab workstation one day in advance.",
            "Project teams may contain at most three students.",
            "API keys must be stored in environment variables.",
            "The project vector database must be Pinecone.",
            "The guide does not specify tuition fees or weekend lab hours.",
        ],
    ),
]


def create_demo(path=Path("examples/course_guide.pdf")):
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setTitle("Northstar Learning Lab - RAG Evaluation Fixture")
    for number, (heading, lines) in enumerate(PAGES, 1):
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(45, 785, heading)
        pdf.setFont("Helvetica", 11)
        for index, line in enumerate(lines):
            pdf.drawString(45, 735 - index * 36, line)
        pdf.setFont("Helvetica", 9)
        pdf.drawString(45, 40, f"Page {number} | Fictional educational example")
        pdf.showPage()
    pdf.save()
    return path


if __name__ == "__main__":
    print(create_demo())
