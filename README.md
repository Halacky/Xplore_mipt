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

