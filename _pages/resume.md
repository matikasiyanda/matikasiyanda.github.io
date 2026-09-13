---
title: "Resume"
permalink: /resume/
redirect_from:
  - /cv/
---

<div class="cv" markdown="1">

<div class="print-only print-header">
  <p class="print-name">Siyanda Matika</p>
  <p class="print-contact">Data and AI Lead · Johannesburg, South Africa · {{ site.author.email }} · linkedin.com/in/siyanda-matika · github.com/{{ site.author.github }} · matikasiyanda.github.io</p>
</div>

Data and AI Lead in Johannesburg, with 8 years of experience across banking,
capital markets, insurance and research. My work spans the full range of data: large structured datasets in SQL, Spark and Hadoop, ETL pipelines,
statistical analysis, and unstructured text, image and video data, through to
machine learning and LLM systems in production. In research, I collaborated on
data pipelines for petabyte-scale radio astronomy data on the path to the Square
Kilometre Array (SKA), as part of international science collaborations.

<p class="cv-actions"><a class="btn-download" href="/assets/files/siyanda-matika-resume.pdf" download>Download PDF</a></p>

<p class="cv-contact"><a href="mailto:{{ site.author.email }}">{{ site.author.email }}</a> · <a href="{{ site.author.linkedin }}">LinkedIn</a> · <a href="https://github.com/{{ site.author.github }}">GitHub</a> · <a href="/work/">Projects</a></p>

## Experience

<div class="cv-role" markdown="1">
### Manager: AI & Data Science <span class="cv-org">Discovery Insure</span>
<p class="cv-when">Aug 2024 – present</p>

- Lead a small data science team working on conversational AI, driver-coaching
  personalisation and the ML platform behind them.
- Building a customer virtual agent for WhatsApp and web: retrieval over
  product documentation, custom guardrail models and drift monitoring.
- Fine-tuned an agent-routing classifier with Azure OpenAI Service, which reads
  the user's intent and routes each conversation to the right agent.
- Took AI systems through privacy and security review: wrote Data Protection
  Impact Assessments (DPIAs), worked with the security team through multiple
  penetration tests, and worked on data governance and adversarial testing of
  chatbots and LLMs.
- Built monitoring on Prometheus and Grafana for model services, with alerts on
  cost, warnings and errors that automatically trigger pipelines.
- Set up Kubeflow pod infrastructure for building agentic systems: agents run in
  isolated Kubernetes pods cut off from the wider network, so an agent that
  executes tools or code can only reach the services it is explicitly allowed
  to use.
- Co-developed a transformer model on sequential health data with Discovery
  Health; presented it at the Discovery Actuarial and Data Science Conference,
  where it placed second.
</div>

<div class="cv-role" markdown="1">
### Machine Learning Engineer <span class="cv-org">Navigare Securities</span>
<p class="cv-when">May 2022 – Aug 2024</p>

- First ML hire: chose the stack, set the roadmap and mentored the engineers
  who joined.
- Ran the GPU infrastructure for model deployments: served and fine-tuned
  open-weight LLMs (Llama, Mixtral), tuned serving for concurrent users, and
  built a RunPod scheduling tool with run/sleep cycles to keep GPU spend bounded.
- Designed the AWS ML stack: serverless model services on ECS and Fargate,
  CI/CD pipelines on GitHub Actions, and data pipelines on Spark and EMR with
  Delta Lake.
- Built retrieval-augmented assistants for capital-markets clients.
- Built Elections Navigator, a public voter-education chatbot for South
  Africa's 2024 national elections that answered voters' questions about
  parties, manifestos and the voting process.
</div>

<div class="cv-role" markdown="1">
### Data Scientist <span class="cv-org">Absa Group</span>
<p class="cv-when">Jan 2020 – May 2022</p>

- Built a customer lifetime value model and a cross-sell recommender for
  merchant devices, both used by marketing and cross-sell teams.
- Built customer spend forecasting and customer-health monitoring, using Spark
  and Hadoop for analytics over the bank's customer data.
- Built Docker containers to deploy the monitoring dashboards.
- Worked with marketing and cross-sell stakeholders to turn model results into
  customer campaigns and value-proposition recommendations, and presented
  findings to senior stakeholders.
</div>

<div class="cv-role" markdown="1">
### Business Analyst <span class="cv-org">Absa Group</span>
<p class="cv-when">Jan 2017 – Jan 2018</p>

- Analysed retail banking data for trends and customer segments.
- Worked directly with product, risk and marketing stakeholders, turning the
  analysis into recommendations for their teams.
</div>

## Skills

<dl class="cv-skills">
  <dt>Data engineering</dt><dd>SQL, Spark, Hadoop, Databricks, Delta Lake; ETL pipelines over large datasets</dd>
  <dt>Statistics and analytics</dt><dd>Bayesian inference, time series, segmentation, forecasting, recommendation, A/B testing</dd>
  <dt>Unstructured data</dt><dd>text, images and video: extraction, OCR, embeddings and retrieval</dd>
  <dt>Machine learning</dt><dd>model design, training, evaluation and fine-tuning across classical ML, deep learning and transformers; PyTorch, scikit-learn, XGBoost</dd>
  <dt>LLMs</dt><dd>retrieval (RAG, hybrid search), fine-tuning (LoRA, Azure OpenAI Service), intent classification and agent routing, reinforcement learning (GRPO), guardrails</dd>
  <dt>Scripting</dt><dd>Python, Bash, SQL</dd>
  <dt>Cloud services</dt><dd>AWS (SageMaker, ECS, Fargate, EMR), Azure (Azure ML, App Service), GCP</dd>
  <dt>GPU infrastructure and deployment</dt><dd>Linux GPU servers (RunPod), open-weight model serving with vLLM, concurrency tuning, Docker</dd>
  <dt>MLOps and LLMOps</dt><dd>deployment and data pipelines, CI/CD (GitHub Actions, Azure DevOps), Kubeflow on Kubernetes, network-isolated pods for agent workloads</dd>
  <dt>Monitoring</dt><dd>Prometheus, Grafana, model and drift monitoring, alerts that trigger pipelines on cost, warnings and errors</dd>
  <dt>Identity and access</dt><dd>IAM on AWS and Azure, Amazon Cognito, Azure App Service Authentication (Easy Auth)</dd>
  <dt>Security and governance</dt><dd>Data Protection Impact Assessments (DPIAs), penetration-test reviews with security teams, adversarial testing of chatbots and LLMs, data governance</dd>
  <dt>Services</dt><dd>FastAPI, Flask</dd>
</dl>

## Education

<div class="cv-role" markdown="1">
### PhD research in Astrophysics, not completed <span class="cv-org">Rhodes University</span>
<p class="cv-when">2018 – 2019</p>

Research on the Epoch of Reionization with the Hydrogen Epoch of Reionization
Array (HERA), a precursor to the Square Kilometre Array (SKA).

- Collaborated on pipelines to process and analyse petabyte-scale radio
  interferometer data, as part of the international HERA collaboration.
- Applied Bayesian inference and Gaussian process regression to separate the
  faint cosmological signal from foregrounds.

Co-author of:

- [Foreground modelling via Gaussian process regression: an application to HERA data](https://doi.org/10.1093/mnras/staa1331),
  *Monthly Notices of the Royal Astronomical Society* 495(3), 2813–2826, 2020
- [Detection of cosmic structures using the bispectrum phase. II. First results from application to cosmic reionization using the Hydrogen Epoch of Reionization Array](https://doi.org/10.1103/PhysRevD.102.022002),
  *Physical Review D* 102, 022002, 2020
</div>

<div class="cv-role" markdown="1">
### MSc Astrophysics and Space Science <span class="cv-org">University of Cape Town and University of the Western Cape</span>
<p class="cv-when">2015 – 2017</p>

Joint UCT–UWC degree through the National Astrophysics and Space Science
Programme (NASSP). Thesis:
[*Detecting Galaxies in the Epoch of Reionization*](https://hdl.handle.net/10566/16687),
supervised by Mario Santos: a Bayesian framework for fitting the 21-cm absorption
feature in radio galaxy spectra, to measure the redshift of galaxies in the
Epoch of Reionization. Also ran SKA1-LOW noise simulations and large-scale
early-universe simulations in C++.
</div>

<div class="cv-role" markdown="1">
### BSc Honours Astrophysics and Space Science <span class="cv-org">University of Cape Town</span>
<p class="cv-when">2014</p>
</div>

<div class="cv-role" markdown="1">
### BSc Physics <span class="cv-org">University of the Western Cape</span>
<p class="cv-when">2011 – 2013</p>
</div>

</div>
