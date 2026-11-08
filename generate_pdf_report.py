import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")
PDF_PATH = os.path.join(BASE_DIR, "reports", "final_report.pdf")

# Custom Canvas to handle running header and "Page X of Y" footer
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#5D6D7E"))

        # Running Header (pages 2 and 3)
        if self._pageNumber > 1:
            self.drawString(36, 762, "Data Management for Machine Learning – Assignment 2 | End-to-End Recommendation Pipeline")
            self.setStrokeColor(colors.HexColor("#D5D8DC"))
            self.setLineWidth(0.5)
            self.line(36, 756, 576, 756)

        # Running Footer (all pages)
        self.setStrokeColor(colors.HexColor("#D5D8DC"))
        self.setLineWidth(0.5)
        self.line(36, 32, 576, 32)
        
        repo_text = "Repository: https://github.com/vv8778/recommendation-pipeline"
        self.drawString(36, 22, repo_text)
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(576, 22, page_str)
        self.restoreState()


def build_pdf():
    # Margins: 36 pt (0.5 inch) left/right, 38 pt top/bottom
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=38,
        bottomMargin=38
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    style_title = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#1B2631"),
        spaceAfter=2
    )

    style_subtitle = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=6
    )

    style_h1 = ParagraphStyle(
        "Heading_1",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=5,
        spaceAfter=3,
        keepWithNext=True
    )

    style_body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.6,
        leading=10.2,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=3
    )

    style_body_bold = ParagraphStyle(
        "BodyBold",
        parent=style_body,
        fontName="Helvetica-Bold"
    )

    style_table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=8.8,
        textColor=colors.HexColor("#2D3748")
    )

    style_table_hdr = ParagraphStyle(
        "TableHdr",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=8.8,
        textColor=colors.HexColor("#FFFFFF")
    )

    story = []

    # =========================================================================
    # PAGE 1: Title, Repo Card, Executive Summary, Architecture, Ingestion & Validation
    # =========================================================================

    # Title & Metadata
    story.append(Paragraph("End-to-End Data Management Pipeline for a Recommendation System", style_title))
    story.append(Paragraph("<b>Course:</b> Data Management for Machine Learning (Assignment 2) &nbsp;|&nbsp; <b>Student ID:</b> vv8778 &nbsp;|&nbsp; <b>Platform Domain:</b> Movie / Media Recommendation", style_subtitle))

    # GitHub Repository Card Table
    repo_card_data = [
        [
            Paragraph("<b>GitHub Submission Repository:</b> &nbsp;<font color='#0969DA'><u>https://github.com/vv8778/recommendation-pipeline</u></font><br/>"
                      "<i>Contains modular source packages, tests, YAML configs, MLflow registry, Airflow DAG, and report artifacts.</i>", style_body)
        ]
    ]
    repo_table = Table(repo_card_data, colWidths=[540])
    repo_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0F4F8")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#B0C4DE")),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(repo_table)
    story.append(Spacer(1, 4))

    # 1. Executive Summary & Design Scope
    story.append(Paragraph("1. Executive Summary & Design Scope", style_h1))
    story.append(Paragraph(
        "Production recommendation systems operate under continuous feedback loops where data quality and feature consistency "
        "dominate model performance. This project implements a fully automated, modular end-to-end data management pipeline for "
        "a movie recommendation platform following the MovieLens schema. The pipeline covers the complete lifecycle: "
        "<b>Data Ingestion &rarr; Validation & Quarantine &rarr; Preparation &rarr; Transformation &rarr; Feature Store &rarr; Model Training &rarr; Evaluation &rarr; Orchestration</b>. "
        "A baseline popularity model and a personalized 20-latent-factor FunkSVD collaborative filtering model were trained with MLflow tracking, "
        "demonstrating a <b>+1,452% lift in Catalog Coverage</b> and a <b>+273% boost in NDCG@10</b> while isolating corrupted data through automated quality gates.",
        style_body
    ))

    # 2. Pipeline Architecture
    story.append(Paragraph("2. Pipeline Architecture & Workflow Flow", style_h1))
    fig1_path = os.path.join(FIG_DIR, "fig1_architecture.png")
    if os.path.exists(fig1_path):
        story.append(Image(fig1_path, width=540, height=135))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "<i>Figure 1: End-to-end data management lifecycle across 7 decoupled sequential stages with dedicated error quarantine and feature stores.</i>",
        ParagraphStyle("Caption", parent=style_body, fontName="Helvetica-Oblique", fontSize=6.8, textColor=colors.HexColor("#5D6D7E"))
    ))

    # 3. Data Ingestion & Quality Validation Strategy
    story.append(Paragraph("3. Data Ingestion & Automated Validation Gate", style_h1))
    story.append(Paragraph(
        "The ingestion module (<code>src/ingestion/ingest.py</code>) processes three core entities: "
        "<b>Users</b> (demographics, age, occupation), <b>Items</b> (titles, multi-label genres, release year), and "
        "<b>Interactions</b> (user_id, item_id, rating, Unix timestamp). Consumption dynamics follow a Pareto power-law user activity distribution "
        "and a Zipfian item popularity distribution to accurately simulate real-world streaming platforms.",
        style_body
    ))
    story.append(Paragraph(
        "Rather than allowing corrupt records to silently pollute training embeddings or trigger pipeline crashes, the automated validation layer "
        "(<code>src/validation/validate.py</code>) enforces five strict quality gates before downstream processing: "
        "<b>1) Missing value checks</b> on critical primary and foreign keys; "
        "<b>2) Deduplication</b> of <code>(user_id, item_id)</code> interaction pairs; "
        "<b>3) Rating boundary checks</b> enforcing ratings &isin; [1.0, 5.0]; "
        "<b>4) Timestamp sanity checks</b> filtering out negative or future timestamps (ts &le; current_time); and "
        "<b>5) Referential integrity checks</b> guaranteeing that every interaction maps to an existing user and catalog item. "
        "Anomalous records are automatically routed to a quarantined error dataset (<code>data/errors/invalid_interactions.csv</code>) "
        "and logged in a machine-readable audit report (<code>reports/data_validation_report.json</code>).",
        style_body
    ))

    # Force Page 2
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: Validation Chart, Preparation, Feature Catalog, Model Training
    # =========================================================================

    # Figure 4: Validation audit chart
    fig4_path = os.path.join(FIG_DIR, "fig4_validation_audit.png")
    if os.path.exists(fig4_path):
        story.append(Image(fig4_path, width=540, height=128))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "<i>Figure 2: Data quality quarantine audit: In an ingested batch of 28,861 interactions, 594 corrupted records (2.06%) were quarantined, passing 28,265 valid records.</i>",
        ParagraphStyle("Caption", parent=style_body, fontName="Helvetica-Oblique", fontSize=6.8, textColor=colors.HexColor("#5D6D7E"))
    ))
    story.append(Spacer(1, 3))

    # 4. Data Preparation & Temporal Train/Test Split
    story.append(Paragraph("4. Data Preparation & Chronological Splitting", style_h1))
    story.append(Paragraph(
        "A critical pitfall in recommendation engineering is random cross-validation, which leaks future user interactions into past training data. "
        "The preparation layer (<code>src/preparation/prepare.py</code>) standardizes ISO 8601 datetimes and executes a <b>per-user chronological time-based split</b>: "
        "the earliest 80% of interactions per user are allocated to <code>train.parquet</code> (23,022 records), while the most recent 20% are reserved in "
        "<code>test.parquet</code> (5,243 records). Furthermore, the transformation layer (<code>src/transformation/transform.py</code>) computes three "
        "analytical datasets on training data: user activity profiles (segmented into casual, moderate, active, and power cohorts), item popularity summaries "
        "with Bayesian shrinkage scores, and a sparse user-item interaction matrix.",
        style_body
    ))

    # 5. Feature Store Management & Catalog
    story.append(Paragraph("5. Feature Store Management & Feature Catalog", style_h1))
    story.append(Paragraph(
        "To eliminate train-serve skew, the Feature Store (<code>src/features/feature_store.py</code>) unifies offline batch feature extraction with an online "
        "key-value lookup API (<code>get_user_features</code>, <code>get_item_features</code>) for real-time recommendation inference.",
        style_body
    ))

    # Feature Catalog Table
    feat_headers = [
        Paragraph("<b>Feature Name</b>", style_table_hdr),
        Paragraph("<b>Entity</b>", style_table_hdr),
        Paragraph("<b>Type</b>", style_table_hdr),
        Paragraph("<b>Calculation Logic</b>", style_table_hdr),
        Paragraph("<b>Reusability Scope</b>", style_table_hdr)
    ]
    feat_rows = [
        [Paragraph("user_avg_rating", style_table_cell), Paragraph("User", style_table_cell), Paragraph("Float", style_table_cell), Paragraph("Historical mean rating given by user", style_table_cell), Paragraph("Offline & Online Serving", style_table_cell)],
        [Paragraph("user_rating_count", style_table_cell), Paragraph("User", style_table_cell), Paragraph("Int", style_table_cell), Paragraph("Total interaction volume per user", style_table_cell), Paragraph("Offline & Online Serving", style_table_cell)],
        [Paragraph("user_recency_days", style_table_cell), Paragraph("User", style_table_cell), Paragraph("Float", style_table_cell), Paragraph("(max_ts - last_user_ts) / 86400", style_table_cell), Paragraph("Offline & Online Serving", style_table_cell)],
        [Paragraph("user_preferred_genre", style_table_cell), Paragraph("User", style_table_cell), Paragraph("String", style_table_cell), Paragraph("Mode genre of user's highest-rated items", style_table_cell), Paragraph("Offline & Online Serving", style_table_cell)],
        [Paragraph("item_bayesian_rating", style_table_cell), Paragraph("Item", style_table_cell), Paragraph("Float", style_table_cell), Paragraph("(v/(v+m))*R + (m/(v+m))*C, m=10", style_table_cell), Paragraph("Offline & Online Serving", style_table_cell)],
        [Paragraph("item_popularity_score", style_table_cell), Paragraph("Item", style_table_cell), Paragraph("Float", style_table_cell), Paragraph("Percentile rank of interaction volume", style_table_cell), Paragraph("Offline & Online Serving", style_table_cell)],
        [Paragraph("genre_* (multi-hot)", style_table_cell), Paragraph("Item", style_table_cell), Paragraph("Vector", style_table_cell), Paragraph("17-dim binary multi-hot genre indicator", style_table_cell), Paragraph("Offline & Online Serving", style_table_cell)],
        [Paragraph("genre_match", style_table_cell), Paragraph("Interaction", style_table_cell), Paragraph("Binary", style_table_cell), Paragraph("1 if preferred genre &isin; item.genres else 0", style_table_cell), Paragraph("Offline Training Matrix", style_table_cell)],
    ]
    t_feat = Table([feat_headers] + feat_rows, colWidths=[95, 55, 45, 205, 140])
    t_feat.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A365D")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F7FAFC")]),
        ("PADDING", (0, 0), (-1, -1), 2.5),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t_feat)
    story.append(Spacer(1, 4))

    # 6. Model Training & MLflow Tracking
    story.append(Paragraph("6. Model Training & MLflow Experiment Tracking", style_h1))
    story.append(Paragraph(
        "Two recommendation paradigms were implemented: "
        "<b>1) Baseline Popularity Recommender:</b> Non-personalized Bayesian shrinkage ranker scoring items by "
        "WR = (v / (v+m)) R + (m / (v+m)) C (where m=10, C=global mean). "
        "<b>2) Personalized Matrix Factorization (FunkSVD):</b> Explicit collaborative filter decomposing ratings into "
        "r_hat(u,i) = &mu; + b_u + b_i + p_u^T q_i with 20 latent factors. "
        "Optimized via Stochastic Gradient Descent (SGD) with L2 regularization (&lambda; = 0.02, &gamma; = 0.005) over 20 epochs. "
        "All parameters, loss trajectories, and serialized binaries (<code>models/*.pkl</code>) were tracked using local MLflow (<code>models/mlflow.db</code>).",
        style_body
    ))

    # Force Page 3
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: Evaluation Results Table, Chart, Orchestration Proof, Limitations
    # =========================================================================

    # 7. Model Evaluation & Benchmark Comparison
    story.append(Paragraph("7. Model Evaluation & Benchmark Results", style_h1))
    story.append(Paragraph(
        "Evaluation was conducted on 5,243 chronologically held-out test interactions across 929 evaluated test users "
        "(relevance threshold r &ge; 4.0) across seven quantitative metrics:",
        style_body
    ))

    # Model Evaluation Table
    eval_headers = [
        Paragraph("<b>Evaluation Metric</b>", style_table_hdr),
        Paragraph("<b>Baseline (Popularity)</b>", style_table_hdr),
        Paragraph("<b>Personalized (FunkSVD)</b>", style_table_hdr),
        Paragraph("<b>Absolute &Delta;</b>", style_table_hdr),
        Paragraph("<b>Relative Lift</b>", style_table_hdr),
        Paragraph("<b>Superior Model</b>", style_table_hdr),
    ]
    eval_rows = [
        [Paragraph("<b>Precision@5</b> &uarr;", style_table_cell), Paragraph("0.0022", style_table_cell), Paragraph("<b>0.0099</b>", style_table_cell), Paragraph("+0.0078", style_table_cell), Paragraph("<b>+360.00%</b>", style_table_cell), Paragraph("Personalized SVD", style_table_cell)],
        [Paragraph("<b>Recall@5</b> &uarr;", style_table_cell), Paragraph("0.0025", style_table_cell), Paragraph("<b>0.0143</b>", style_table_cell), Paragraph("+0.0118", style_table_cell), Paragraph("<b>+477.63%</b>", style_table_cell), Paragraph("Personalized SVD", style_table_cell)],
        [Paragraph("<b>NDCG@5</b> &uarr;", style_table_cell), Paragraph("0.0023", style_table_cell), Paragraph("<b>0.0143</b>", style_table_cell), Paragraph("+0.0120", style_table_cell), Paragraph("<b>+524.38%</b>", style_table_cell), Paragraph("Personalized SVD", style_table_cell)],
        [Paragraph("<b>Precision@10</b> &uarr;", style_table_cell), Paragraph("0.0032", style_table_cell), Paragraph("<b>0.0082</b>", style_table_cell), Paragraph("+0.0050", style_table_cell), Paragraph("<b>+153.33%</b>", style_table_cell), Paragraph("Personalized SVD", style_table_cell)],
        [Paragraph("<b>Recall@10</b> &uarr;", style_table_cell), Paragraph("0.0080", style_table_cell), Paragraph("<b>0.0243</b>", style_table_cell), Paragraph("+0.0163", style_table_cell), Paragraph("<b>+205.24%</b>", style_table_cell), Paragraph("Personalized SVD", style_table_cell)],
        [Paragraph("<b>NDCG@10</b> &uarr;", style_table_cell), Paragraph("0.0047", style_table_cell), Paragraph("<b>0.0177</b>", style_table_cell), Paragraph("+0.0129", style_table_cell), Paragraph("<b>+273.63%</b>", style_table_cell), Paragraph("Personalized SVD", style_table_cell)],
        [Paragraph("<b>Catalog Coverage@10</b> &uarr;", style_table_cell), Paragraph("0.0460 (4.6%)", style_table_cell), Paragraph("<b>0.7140 (71.4%)</b>", style_table_cell), Paragraph("+0.6680", style_table_cell), Paragraph("<b>+1,452.17%</b>", style_table_cell), Paragraph("Personalized SVD", style_table_cell)],
        [Paragraph("<b>Intra-List Diversity@10</b> &uarr;", style_table_cell), Paragraph("0.9223", style_table_cell), Paragraph("<b>0.9279</b>", style_table_cell), Paragraph("+0.0056", style_table_cell), Paragraph("+0.61%", style_table_cell), Paragraph("Personalized SVD", style_table_cell)],
        [Paragraph("<b>RMSE (Rating Error)</b> &darr;", style_table_cell), Paragraph("<b>1.1295</b>", style_table_cell), Paragraph("1.1867", style_table_cell), Paragraph("+0.0572", style_table_cell), Paragraph("-5.07%", style_table_cell), Paragraph("Baseline", style_table_cell)],
        [Paragraph("<b>MAE (Rating Error)</b> &darr;", style_table_cell), Paragraph("<b>0.9445</b>", style_table_cell), Paragraph("0.9704", style_table_cell), Paragraph("+0.0259", style_table_cell), Paragraph("-2.74%", style_table_cell), Paragraph("Baseline", style_table_cell)],
    ]
    t_eval = Table([eval_headers] + eval_rows, colWidths=[120, 85, 95, 75, 80, 85])
    t_eval.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A365D")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F7FAFC")]),
        ("PADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    story.append(t_eval)
    story.append(Spacer(1, 2))

    # Figure 3: Model comparison chart
    fig3_path = os.path.join(FIG_DIR, "fig3_model_comparison.png")
    if os.path.exists(fig3_path):
        story.append(Image(fig3_path, width=540, height=128))
    story.append(Spacer(1, 2))

    # 8. Orchestration & Reproducibility Proof
    story.append(Paragraph("8. Workflow Orchestration & Execution Proof", style_h1))
    story.append(Paragraph(
        "The workflow is automated via an Apache Airflow DAG (<code>dags/airflow_recommendation_dag.py</code>) with task sequence: "
        "<code>ingest &gt;&gt; validate &gt;&gt; prepare &gt;&gt; transform &gt;&gt; features &gt;&gt; train &gt;&gt; evaluate</code>. "
        "A reproducible CLI runner (<code>python -m src.orchestration.pipeline</code>) executes the complete pipeline in <b>21.20 seconds</b>:",
        style_body
    ))

    # Figure 2: Terminal run screenshot
    fig2_path = os.path.join(FIG_DIR, "fig2_terminal_run.png")
    if os.path.exists(fig2_path):
        story.append(Image(fig2_path, width=540, height=132))
    story.append(Spacer(1, 2))

    # 9. Production Limitations & Design Trade-offs
    story.append(Paragraph("9. Production Limitations & Scalability Trade-offs", style_h1))
    story.append(Paragraph(
        "<b>1) Day-Zero Cold-Start:</b> Pure matrix factorization struggles with unrated new items or users. In production, a two-tower neural architecture "
        "incorporating item text embeddings (e.g. Sentence-BERT) is recommended. "
        "<b>2) Scaling Latency:</b> While in-memory SGD completes in 15s for 28k interactions, scaling to millions requires distributed ALS (PySpark/Implicit) "
        "and Approximate Nearest Neighbor vector search (FAISS/Milvus). "
        "<b>3) Implicit Feedback:</b> Explicit ratings are often sparse; adapting loss formulations to Bayesian Personalized Ranking (BPR) is optimal for clickstream logs.",
        style_body
    ))

    # Build document with custom canvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Final report PDF successfully generated at: {PDF_PATH}")

if __name__ == "__main__":
    build_pdf()
