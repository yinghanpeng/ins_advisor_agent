.PHONY: db-upgrade db-downgrade db-reset-local test

db-upgrade:
	python3 scripts/db_upgrade.py

db-downgrade:
	python3 scripts/db_downgrade.py

db-reset-local: db-downgrade db-upgrade

test:
	python3 -m pytest
