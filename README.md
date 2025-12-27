# Smart Screening Engine

## How to deploy

Go to the monitoring directory:

```
cd Xplore_mipt/monitoring
```

after that, run the container build.
```
docker compose up -d
```

After launching, you will receive:

1. web UI interface available at http://localhost:3000/

2. Prometheus is available at http://localhost:9090/

3. Grafana at http://localhost:3001/


Also, before launching, create an .env file (an example of filling can be found in the .env.example file). Do not forget to replace the parameters with yours (the address of the provider providing hosting services for LLM models, the token and the names of the models according to your provider's api)

Work results:

https://github.com/user-attachments/assets/e99de96e-91b3-4ef1-b60b-177e57218e5b

Metrics:

<img width="761" height="480" alt="Screenshot from 2025-12-24 21-01-42" src="https://github.com/user-attachments/assets/9ca97bc3-7282-4ad2-8d5b-6597a815f338" />

We assessed the project very rigorously, so the metrics aren't particularly good. However, a preliminary analysis reveals a significant question regarding the true "expert_eligibility" values. In cases where there are discrepancies between the expert assessment and our model's assessment, common sense dictates that the explanation provided by our model is very sound, which calls into question the expert assessment. For example, the source data lack information about signed consent, yet the expert still states that the patient should be included in the study (even though signed consent is a mandatory inclusion criterion). There are also discrepancies with threshold values. For example, for patient emp-013, the LVEF is listed as 39%. The criteria specify a range of EF ≥ 36% to ≤ 40%. For an EF of 36–40%, the following requirements apply: Without (AF): NT-proBNP ≥ 2500 pg/mL, with AF: NT-proBNP ≥ 5000 pg/mL. In the patient: NT-proBNP = 860 pg/mL - which does not meet any of the thresholds. However, the labeling is still included.

Overall, the resulting solution is useful, if not for decision making, then at least for automating labeling and validating existing labeling.
