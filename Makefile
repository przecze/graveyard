.PHONY: deploy down logs build lockfile

lockfile:
	docker run --rm -v "$$(pwd)/frontend":/app -w /app node:24-alpine npm install

# Deploy to production server
deploy:
	cd ansible && ansible-playbook deploy.yml

# Deploy with dry-run (shows what would happen)
deploy-check:
	cd ansible && ansible-playbook deploy.yml --check --diff
