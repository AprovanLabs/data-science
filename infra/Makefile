run:
	kubectl ctx rancher-desktop
	kustomize build . | kubectl apply -f -

delete:
	kustomize build . | kubectl delete -f -

db:
	kubectl port-forward service/postgres 5432
