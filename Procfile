web: flask db upgrade; flask translate compile; gunicorn InvestigationApp:app
worker: rq worker InvestigationApp-tasks
