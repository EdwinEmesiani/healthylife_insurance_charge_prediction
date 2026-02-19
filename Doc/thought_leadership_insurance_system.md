

## Thought Leadership Article


**Title:**  
**From EDA to Deployed Insurance Models: Thinking Like an AI Systems Architect**

Modern AI work is shifting from “I trained a model” to “I built a system that delivers value.”  
In my recent projects, I deliberately structured my work to reflect this shift.

### Stage 1 – Establishing the Analytics Foundation (Technoecom E-Commerce EDA)

For Technoecom, a fictional e-commerce company, I started where every serious AI system should start: **with questions, not models**.

- Which warehouses actually drive revenue?
- How seasonal is demand?
- Are coupons increasing long-term satisfaction or just subsidising one-off orders?

The deliverable was not just a notebook full of plots; it was a **CXO-oriented analytics artefact**, exportable to HTML/PDF, designed for decision makers. This EDA layer de-risks future modelling because it creates a shared understanding of the business and the data.

### Stage 2 – Building a Deployable ML System (HealthyLife Insurance)

For HealthyLife Insurance, a New York-based insurance provider, the goal was to predict **health insurance charges** from demographic and behavioural attributes.

Instead of stopping at “I fit a regression model”, I treated the project as a **full system**:

- **Preprocessing as a first-class citizen**  
  Using `ColumnTransformer` and Pipelines, I ensured that feature scaling and encoding were defined once and used consistently for both training and inference.

- **Multiple models, single interface**  
  Linear Regression, Decision Trees, and Random Forests were all wrapped in the same pipeline interface, allowing for clean evaluation and future swaps.

- **Deployment as a first-class concern**  
  The trained pipeline was exported as `model.joblib` and served via a **Gradio application** deployed to **Hugging Face Spaces**. Non-technical stakeholders can now explore the model’s behaviour directly.

### Stage 3 – Monitoring and Drift Awareness

Real systems don’t end when the model goes live.

To reflect this, I introduced:

- **Request + prediction logging** via JSON lines and `CommitScheduler` into a Hugging Face dataset.
- **Model drift checks**, comparing training target distributions to live prediction distributions.
- **Data drift indicators**, monitoring whether the profile of incoming customers (e.g. age, BMI, smoker rates) deviates from the training population.

This creates a pathway for **retriggers of retraining**, even if the initial implementation uses simple thresholds.

### Stage 4 – Toward CI/CD for ML

The natural next step is to hook everything into an automated workflow:

- When new training data is available, or drift exceeds a threshold:
  - A GitHub Action can run `train.py`,
  - Push an updated `model.joblib`,
  - And trigger a redeployment of the Space.

At that point, we’re no longer talking about “toy projects”. We’re talking about **AI systems that live, adapt, and stay aligned with the data they serve.**

### Closing Thought

The difference between a portfolio that says “I know ML” and one that says “I architect AI systems” lies in precisely these steps:

- EDA that serves the business,
- Pipelines that serve deployment,
- Monitoring that serves reliability,
- And automation that serves evolution.

That is the standard I aim for in my GitHub work: **engineer building deployable AI systems**, not just models.
