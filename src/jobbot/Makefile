include .env
export

scrape:
	mkdir -p target \
		&& cat catalog.yml | envsubst \
		> target/catalog.yml

	python3 jobbot_cli.py scrape \
		-f target/catalog.yml \
		-t catagory:jobs location:minnesota

clean:
	rm -rf target/candidates
	rm -rf target/job_history
