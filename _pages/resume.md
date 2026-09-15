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
capital markets, insurance and research. I take data and AI work from the
first scoping conversation to a system running in production, and I have
done it at both ends of the scale: in research, on petabytes of
radio-interferometer data as part of the international HERA collaboration on
the path to the Square Kilometre Array, where the pipelines had to be right
because the signal being searched for was fainter than everything in front
of it; and in industry, on bank, capital-markets and insurance data, where
the same discipline applies to statistical models, retrieval systems and
language models that customers use. Structured and unstructured data alike,
with the evaluation to say whether a thing works.

<p class="cv-actions"><a class="btn-download" href="/assets/files/siyanda-matika-resume.pdf" download>Download PDF</a></p>

<p class="cv-contact"><a href="mailto:{{ site.author.email }}">{{ site.author.email }}</a> · <a href="{{ site.author.linkedin }}">LinkedIn</a> · <a href="https://github.com/{{ site.author.github }}">GitHub</a> · <a href="/work/">Projects</a></p>

## Experience

<div class="cv-role" markdown="1">
### Manager: AI & Data Science <span class="cv-org">Discovery Insure</span>
<p class="cv-when">Aug 2024 – present</p>

- Own AI delivery across conversational AI, driver-coaching personalisation and
  the ML platform behind them. Manage the data science team and run technical
  hiring for the function.
- Lead the customer virtual agent for WhatsApp and web end to end: scope, agent
  architecture, retrieval over product documentation, guardrail strategy and
  integration into customer channels.
- Own the agent-routing classifier, fine-tuned on Azure OpenAI Service to read
  each user's intent and route the conversation to the right agent.
- Accountable for taking AI systems through privacy and security sign-off:
  authored the Data Protection Impact Assessments (DPIAs), coordinated multiple
  penetration tests with the security team, and set up data governance and
  adversarial testing for chatbots and LLMs.
- Established monitoring for model services on Prometheus and Grafana, with
  alerts on cost, warnings and errors that automatically trigger pipelines.
- Designed the Kubeflow pod infrastructure for agent development: agents run in
  isolated Kubernetes pods cut off from the wider network, so an agent that
  executes tools or code can only reach the services it is explicitly allowed
  to use.
- Defined the driver-coaching personalisation roadmap, moving from one fixed
  reward structure for every driver to coaching that adapts to each driver's
  behaviour.
- Co-led a transformer model on sequential health data with Discovery Health;
  presented it at the Discovery Actuarial and Data Science Conference, where it
  placed second.
</div>

<div class="cv-role" markdown="1">
### Machine Learning Engineer <span class="cv-org">Navigare Securities</span>
<p class="cv-when">May 2022 – Aug 2024</p>

- Joined as the first ML hire and worked at lead level: chose the stack, set the
  roadmap, and managed the AI product portfolio for capital-markets clients.
- Sat on the hiring panel for the ML team: interviewed and hired three machine
  learning engineers, then managed and mentored them.
- Owned GPU infrastructure and model deployment: served and fine-tuned
  open-weight LLMs (Llama, Mistral, Qwen), tuned serving for concurrent
  users, and introduced RunPod scheduling with run/sleep cycles to keep GPU
  spend bounded.
- Designed and ran the AWS ML stack, provisioned with Terraform: serverless
  model services on ECS and Fargate, CI/CD pipelines on GitHub Actions, and
  data pipelines on Spark and EMR with Delta Lake.
- Led conversational AI for capital-markets clients, including the decision to
  design an in-house hybrid retrieval framework when off-the-shelf retrieval
  fell short on precision.
- Conceived and led Elections Navigator, a public voter-education chatbot for
  South Africa's 2024 national elections, from scope to launch against the
  fixed election deadline.
</div>

<div class="cv-role" markdown="1">
### Data Scientist <span class="cv-org">Absa Group</span>
<p class="cv-when">Jan 2020 – May 2022</p>

- Owned the customer lifetime value model and the merchant-device cross-sell
  recommender from development to production; both were adopted by the
  marketing and cross-sell teams.
- Delivered customer spend forecasting and customer-health monitoring on Spark
  and Hadoop over the bank's customer data, and deployed the monitoring
  dashboards in Docker containers.
- Managed engagement with marketing and cross-sell stakeholders, turning model
  results into campaign and value-proposition recommendations presented to
  senior stakeholders.
- Interviewed and hired data scientists for the team, then mentored them and
  managed their day-to-day work.
</div>

<div class="cv-role" markdown="1">
### Business Analyst <span class="cv-org">Absa Group</span>
<p class="cv-when">Jan 2017 – Jan 2018</p>

- Owned analyses of retail banking data on trends and customer segments.
- Managed engagement with product, risk and marketing stakeholders, turning
  the analysis into recommendations for their teams.
</div>

## Skills

<dl class="cv-skills">
  <dt>Data engineering</dt><dd>Pipelines from raw source to modelling table, at two scales: petabytes of radio-interferometer data in international science collaborations, and bank-scale customer data in Spark, Hadoop and Databricks with Delta Lake for versioned tables. SQL is the working language; Python and Bash do the rest.</dd>
  <dt>Statistics and modelling</dt><dd>Bayesian inference, time-series forecasting, segmentation and recommendation, and A/B tests designed for the power they need and read with the error bars they deserve. Classical machine learning through deep learning in PyTorch, scikit-learn and XGBoost, with evaluation that separates a real gain from single-seed noise.</dd>
  <dt>Language models</dt><dd>Open-weight models (Llama, Mistral, Qwen) and Azure OpenAI in production. Hybrid retrieval built and measured end to end, fine-tuning with LoRA, reinforcement learning with GRPO, intent classification and agent routing, and guardrails tested adversarially before anything reaches a customer.</dd>
  <dt>Unstructured data</dt><dd>Text, images and video: extraction and OCR, embeddings, and retrieval systems evaluated against held-out sets rather than demos.</dd>
  <dt>Infrastructure and deployment</dt><dd>AWS (SageMaker, ECS, Fargate, EMR), Azure (Azure ML, App Service) and GCP, provisioned with Terraform. Kubeflow on Kubernetes with network-isolated pods for agent workloads. GPU serving with vLLM on Linux servers and RunPod, tuned for concurrent users and scheduled to keep spend bounded. Docker, FastAPI and Flask services, CI/CD on GitHub Actions and Azure DevOps.</dd>
  <dt>Operations and monitoring</dt><dd>Prometheus and Grafana for services, model and drift monitoring for what they serve, and alerts wired to pipelines so that cost, warnings and errors trigger an action rather than an email.</dd>
  <dt>Security, identity and governance</dt><dd>IAM on AWS and Azure, Amazon Cognito and App Service authentication. Data Protection Impact Assessments, penetration-test reviews with security teams, adversarial testing of chatbots and language models, and data governance.</dd>
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
