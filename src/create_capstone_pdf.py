from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.lib.units import inch
from pathlib import Path

ROOT = Path.cwd()
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

PDF = REPORTS / "data_science_capstone_whitepaper.pdf"
MD = REPORTS / "data_science_capstone_whitepaper.md"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="CapTitle", parent=styles["Title"], fontSize=22, leading=27,
    alignment=TA_CENTER, spaceAfter=20
))
styles.add(ParagraphStyle(
    name="CapSub", parent=styles["Normal"], fontSize=11, leading=16,
    alignment=TA_CENTER, spaceAfter=12
))
styles.add(ParagraphStyle(
    name="CapH1", parent=styles["Heading1"], fontSize=14, leading=18,
    spaceBefore=10, spaceAfter=7
))
styles.add(ParagraphStyle(
    name="CapH2", parent=styles["Heading2"], fontSize=11, leading=15,
    spaceBefore=7, spaceAfter=4
))
styles.add(ParagraphStyle(
    name="CapBody", parent=styles["BodyText"], fontSize=9.2,
    leading=13.5, alignment=TA_JUSTIFY, spaceAfter=6
))
styles.add(ParagraphStyle(
    name="CapSmall", parent=styles["BodyText"], fontSize=7.8,
    leading=10
))

story = []

def title(text):
    story.append(Paragraph(text, styles["CapTitle"]))

def h(text):
    story.append(Paragraph(text, styles["CapH1"]))

def h2(text):
    story.append(Paragraph(text, styles["CapH2"]))

def p(text):
    story.append(Paragraph(text, styles["CapBody"]))

# TITLE PAGE
story.append(Spacer(1, 1.0*inch))
title("DATA SCIENCE RESEARCH CAPSTONE")
story.append(Paragraph(
    "An End-to-End Reproducible Data Science Framework for Statistical Inference, "
    "Unsupervised Learning, and Time-Series Forecasting",
    styles["CapSub"]
))
story.append(Spacer(1, 0.25*inch))
story.append(Paragraph("Formal Research Whitepaper", styles["CapSub"]))
story.append(Spacer(1, 1.25*inch))
story.append(Paragraph(
    "Statistical Analysis • PCA • Unsupervised Clustering • ARIMA Forecasting",
    styles["CapSub"]
))
story.append(Spacer(1, 1.5*inch))
story.append(Paragraph("September 2026", styles["CapSub"]))
story.append(PageBreak())

# ABSTRACT
h("Abstract")
p(
"This capstone synthesizes a reproducible data science workflow spanning statistical "
"inference, dimensionality reduction, unsupervised learning, and time-series forecasting. "
"The research repository contains executed notebooks, source utilities, tests, datasets, "
"reports, figures, and configuration artifacts. The statistical component evaluates "
"distributional assumptions and group differences using Shapiro–Wilk, Kolmogorov–Smirnov, "
"Welch's t-test, Mann–Whitney U, one-way ANOVA, two-way ANOVA, and Tukey HSD. The "
"unsupervised-learning component applies principal component analysis (PCA) and compares "
"K-Means, DBSCAN, and hierarchical clustering using internal validation. The forecasting "
"component applies ARIMA model selection to a supplied daily demand series. The demand "
"dataset contains 30 observations from 1 June through 30 June 2026; consequently, a "
"15-day training/15-day holdout design was used, followed by a 30-day forward forecast. "
"The repository also distinguishes synthetic demonstration datasets from the supplied "
"demand dataset. The central contribution is methodological: the project demonstrates "
"how analytical assumptions, model evaluation, documentation, and reproducibility can "
"be integrated into one auditable data science workflow."
)

h("1. Introduction")
p(
"Modern data science requires more than producing a model output. A defensible analytical "
"project must connect a research question to data preparation, assumptions, statistical "
"methods, model selection, evaluation, interpretation, and reproducibility. This capstone "
"brings those elements together in a single repository."
)
p(
"The project intentionally uses several complementary analytical techniques. Statistical "
"inference addresses questions about distributions and group differences. PCA reduces "
"dimensionality and provides a compact representation of multivariate structure. Clustering "
"explores latent groups without requiring pre-existing labels. Time-series forecasting "
"addresses the separate problem of estimating future demand from ordered observations."
)

h("2. Research Question and Objectives")
p(
"<b>Research Question:</b> How can a reproducible data science workflow combine statistical "
"inference, unsupervised learning, and time-series forecasting to identify meaningful "
"structure and support evidence-based forecasting decisions?"
)
p(
"<b>Objectives:</b> (1) evaluate statistical assumptions and group differences; "
"(2) reduce dimensionality and examine latent structure using PCA; "
"(3) compare multiple clustering algorithms using internal validation; "
"(4) develop and evaluate an ARIMA forecasting workflow; and "
"(5) document the analysis so that the computational process can be reproduced and audited."
)

h("3. Literature Background")
p(
"Principal component analysis is a standard multivariate technique for transforming correlated "
"variables into orthogonal components ordered by explained variance. Jolliffe and Cadima "
"describe PCA as a widely used method for dimensionality reduction and exploratory analysis. "
"Clustering methods such as K-Means, density-based clustering, and hierarchical clustering "
"represent different assumptions about the geometry and organization of observations."
)
p(
"For time-series analysis, the ARIMA framework combines autoregressive and moving-average "
"components with differencing to address non-stationary series. Box, Jenkins, Reinsel, and "
"Ljung provide the classical framework for ARIMA modeling and forecasting. Modern forecasting "
"practice also emphasizes separating model selection from out-of-sample evaluation."
)
p(
"Reproducible computational research requires that analytical environments, code, data, "
"and computational outputs be organized so that results can be independently regenerated. "
"Best-practice literature emphasizes version control, automated testing, documentation, "
"and preservation of computational provenance."
)

h("4. Data Sources and Governance")
p(
"The repository contains multiple datasets associated with different analytical tasks. "
"The daily demand series is the supplied forecasting dataset with the columns date, demand, "
"marketing_event, and holiday. It contains 30 daily observations covering 1 June 2026 "
"through 30 June 2026."
)
p(
"The statistical-analysis and clustering components use synthetic or demonstration datasets. "
"These datasets are useful for demonstrating analytical procedures, but their results must "
"not be interpreted as empirical evidence about a real external population. This distinction "
"is maintained throughout the capstone to preserve research integrity."
)

h("5. Analytical Methodology")

h2("5.1 Statistical Hypothesis Testing")
p(
"The statistical workflow begins with distributional diagnostics. Shapiro–Wilk and "
"Kolmogorov–Smirnov tests are used to evaluate distributional assumptions. For two-group "
"comparisons, Welch's t-test provides a variance-robust parametric comparison, while "
"Mann–Whitney U provides a non-parametric alternative. One-way and two-way ANOVA are used "
"for multi-factor mean comparisons, with Tukey HSD providing post-hoc pairwise analysis "
"where appropriate. Confidence intervals complement p-values by communicating uncertainty."
)

h2("5.2 Dimensionality Reduction")
p(
"PCA transforms the original feature space into principal components. The executed notebook "
"reports component-level explained variance and cumulative explained variance and visualizes "
"the scree/cumulative relationship. This allows the analysis to identify how much information "
"is represented by progressively larger numbers of components."
)

h2("5.3 Unsupervised Clustering")
p(
"K-Means, DBSCAN, and hierarchical clustering are evaluated as complementary approaches. "
"K-Means is examined through an elbow analysis and silhouette-based validation. DBSCAN "
"provides a density-based alternative, while hierarchical clustering provides a tree-based "
"view of similarity. Internal metrics are used as diagnostics rather than as proof of "
"business usefulness."
)

h2("5.4 ARIMA Forecasting")
p(
"The demand series is ordered chronologically and evaluated within an ARIMA modeling workflow. "
"Stationarity is investigated using the Augmented Dickey–Fuller framework. Multiple candidate "
"ARIMA orders are compared using AIC, after which the selected specification is evaluated on "
"a holdout period. Because only 30 observations are available, the first 15 observations are "
"used for training and the final 15 observations for holdout evaluation. A separate 30-day "
"forecast is then generated beginning 1 July 2026."
)

h("6. Statistical Analysis Results")
p(
"The executed statistical notebook contains saved outputs for Shapiro–Wilk and "
"Kolmogorov–Smirnov diagnostics, Welch's t-test, Mann–Whitney U, one-way ANOVA, "
"two-way ANOVA, Tukey HSD, confidence intervals, and supporting visualizations. "
"The results demonstrate a complete hypothesis-testing workflow rather than isolated "
"individual tests."
)
p(
"Interpretation is constrained by the nature of the demonstration data. The statistical "
"results are evidence about the analyzed synthetic dataset and should not be generalized "
"to an external population without additional real-world data and study design."
)

h("7. PCA and Clustering Results")
p(
"The executed PCA/clustering notebook contains the full explained-variance table, "
"scree/cumulative-variance visualization, K-Means elbow analysis, silhouette evaluation, "
"and comparative visualizations for K-Means, DBSCAN, and hierarchical clustering."
)
p(
"The analysis demonstrates why clustering should be treated as an exploratory modeling task. "
"Different algorithms can produce different structures because they encode different assumptions "
"about cluster geometry and density. Silhouette scores and elbow diagnostics provide useful "
"internal evidence, but cluster stability, domain interpretation, and external validation "
"would be required before operational deployment."
)

h("8. Time-Series Forecasting Results")
p(
"The forecasting dataset contains 30 daily demand observations from 1 June through 30 June "
"2026. The workflow generates historical-demand and decomposition visualizations, evaluates "
"a holdout forecast, and produces a 30-day future forecast."
)
p(
"The first five forecast values for the future horizon are approximately: 1 July 2026 = "
"155.94; 2 July = 158.02; 3 July = 152.99; 4 July = 149.84; and 5 July = 151.57 demand units."
)
p(
"The forecasting artifacts include a 30-day forecast CSV and forecast visualizations. "
"Because the underlying series is short, the forecast should be regarded as a reproducible "
"modeling demonstration rather than a production-grade long-range planning estimate."
)

h("9. Model Performance and Evaluation")
p(
"Model selection and model evaluation are deliberately treated as separate activities. "
"For ARIMA, AIC supports comparison among candidate specifications, while holdout evaluation "
"provides an out-of-sample check. For clustering, silhouette and elbow diagnostics provide "
"internal validation. For statistical inference, p-values and confidence intervals are "
"interpreted alongside assumptions and study design."
)
p(
"The detailed numerical outputs remain preserved in the executed notebooks and generated "
"reports. Keeping the notebooks as the computational source of truth reduces transcription "
"risk when the capstone narrative is condensed for publication."
)

h("10. Practical and Business Implications")
p(
"An integrated analytical workflow can support several practical decision contexts. Statistical "
"testing can identify whether observed group differences merit further investigation. PCA can "
"simplify complex feature spaces for exploratory analysis. Clustering can support segmentation "
"when labeled outcomes are unavailable. Forecasting can provide a quantitative baseline for "
"short-term demand planning."
)
p(
"In an operational environment, these methods should be connected to domain-specific costs, "
"uncertainty, monitoring, and governance. Forecast intervals and scenario analysis would be "
"particularly important when forecasts influence inventory, staffing, capacity, or resource "
"allocation decisions."
)

h("11. Limitations")
p(
"The most important limitation is data scope. The supplied forecasting series contains only "
"30 observations, which restricts the reliability of long-horizon forecasting and limits the "
"ability to identify recurring seasonal patterns. The clustering and statistical datasets are "
"synthetic demonstrations and therefore cannot establish external validity."
)
p(
"Internal clustering metrics also do not demonstrate that a segmentation is commercially useful. "
"Statistical tests may be affected by sample size, assumptions, multiple comparisons, and study "
"design. Future analyses should therefore use larger real-world datasets and stronger validation "
"strategies."
)

h("12. Reproducibility and Quality Assurance")
p(
"The repository follows a structured organization separating raw, interim, and processed data, "
"notebooks, source code, tests, reports, and preregistration materials. Executed notebooks "
"preserve computational outputs, while generated reports and figures provide an audit trail "
"between analysis and communication."
)
p(
"Repository inspection also identified several empty placeholder files, including README.md, "
"reports/final_report.md, preregistration/analysis_preregistration.md, and src/analysis.py. "
"These are documented as repository-state observations and are not represented as completed "
"analytical artifacts."
)

h("13. Future Research")
p(
"Future work should prioritize larger real-world datasets, rolling-origin time-series validation, "
"prediction intervals, external regressors, seasonal models where justified, cluster stability "
"analysis, sensitivity analysis, automated testing, and a fully populated preregistration. "
"These additions would improve both scientific validity and operational usefulness."
)

h("14. Conclusion")
p(
"This capstone demonstrates an end-to-end reproducible data science framework connecting "
"research questions, statistical inference, dimensionality reduction, unsupervised learning, "
"forecasting, evaluation, documentation, and quality assurance. Its principal contribution "
"is methodological rather than a claim of universal predictive superiority."
)
p(
"The supplied demand series provides a concrete forecasting example, while the statistical "
"and clustering components demonstrate complementary analytical capabilities on clearly "
"identified synthetic datasets. The resulting repository provides a transparent foundation "
"for extending the work to larger, validated, domain-specific research datasets."
)

h("References")
refs = [
"Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time Series Analysis: Forecasting and Control (5th ed.). Wiley.",
"Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: a review and recent developments. Philosophical Transactions of the Royal Society A, 374.",
"Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality (complete samples). Biometrika, 52(3–4), 591–611.",
"Sandve, G. K., Nekrutenko, A., Taylor, J., & Hovig, E. (2013). Ten simple rules for reproducible computational research. PLoS Computational Biology, 9(10).",
"Wilson, G., et al. (2014). Best practices for scientific computing. PLoS Biology, 12(1).",
"MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability.",
"Hyndman, R. J., & Athanasopoulos, G. Forecasting: Principles and Practice. OTexts."
]
for r in refs:
    p(r)

h("Appendix A — Repository Evidence")
inventory = [
["Artifact","Purpose","Status"],
["daily-demand-series.csv","Supplied forecasting dataset","Executed"],
["advanced_statistical_analysis.ipynb","Statistical hypothesis testing","Executed"],
["dimensionality_reduction_clustering.ipynb","PCA and clustering","Executed"],
["time_series_forecasting_arima.ipynb","ARIMA forecasting","Executed"],
["reports/","Reports, figures and forecast output","Present"],
["src/","Supporting utilities","Present"],
["tests/","Testing/reproducibility support","Present"],
["README.md","Repository documentation","Placeholder"],
["preregistration/","Research planning artifacts","Placeholder"],
]
table = Table(inventory, colWidths=[2.35*inch,3.05*inch,1.25*inch], repeatRows=1)
table.setStyle(TableStyle([
("BACKGROUND",(0,0),(-1,0),colors.HexColor("#263238")),
("TEXTCOLOR",(0,0),(-1,0),colors.white),
("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
("FONTSIZE",(0,0),(-1,-1),7.4),
("GRID",(0,0),(-1,-1),0.35,colors.grey),
("VALIGN",(0,0),(-1,-1),"TOP"),
("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F3F5F6")]),
("LEFTPADDING",(0,0),(-1,-1),5),
("RIGHTPADDING",(0,0),(-1,-1),5),
]))
story.append(table)

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica",7.5)
    canvas.drawString(.65*inch,.42*inch,"Data Science Research Capstone")
    canvas.drawRightString(7.85*inch,.42*inch,"Page %d" % doc.page)
    canvas.restoreState()

doc = SimpleDocTemplate(
    str(PDF), pagesize=A4,
    rightMargin=.65*inch, leftMargin=.65*inch,
    topMargin=.65*inch, bottomMargin=.62*inch
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)

MD.write_text("""# Data Science Research Capstone

## An End-to-End Reproducible Data Science Framework for Statistical Inference, Unsupervised Learning, and Time-Series Forecasting

### Abstract
This capstone synthesizes a reproducible workflow spanning statistical inference, PCA, unsupervised clustering, and ARIMA forecasting. The repository contains executed notebooks, source utilities, tests, reports, figures, and datasets. The supplied demand dataset contains 30 daily observations from 1 June through 30 June 2026. A 15-day training/15-day holdout design was therefore used, followed by a 30-day forecast beginning 1 July 2026.

### Research Question
How can a reproducible data science workflow combine statistical inference, unsupervised learning, and time-series forecasting to identify meaningful structure and support evidence-based forecasting decisions?

### Methods
Statistical analysis includes Shapiro–Wilk, Kolmogorov–Smirnov, Welch's t-test, Mann–Whitney U, one-way ANOVA, two-way ANOVA, and Tukey HSD. PCA is used for dimensionality reduction. K-Means, DBSCAN, and hierarchical clustering are compared using internal validation. ARIMA candidate models are evaluated using AIC and a holdout period.

### Forecasting Result
The first five forecast values are approximately 155.94, 158.02, 152.99, 149.84, and 151.57 for 1–5 July 2026.

### Research Integrity
The statistical and clustering datasets are synthetic/demo datasets. The demand series is the supplied forecasting dataset. Synthetic results are not presented as evidence about an external population.

### Limitations
The demand series is short, limiting long-horizon forecasting reliability. Internal clustering metrics do not establish business usefulness, and synthetic analyses do not establish external validity.

### Conclusion
The capstone demonstrates an end-to-end reproducible framework connecting research questions, statistical inference, unsupervised learning, forecasting, evaluation, and documentation.
""", encoding="utf-8")

print("")
print("==============================================")
print("CAPSTONE PDF CREATED SUCCESSFULLY")
print("==============================================")
print("PDF :", PDF)
print("MD  :", MD)
print("")
