# Local Data Science

## Pre-requisites

```bash
$> brew install kustomize skaffold
$> brew install --cask rancher lens
```

## Deploy

```bash
infra/k8s$> kubectl ctx rancher-desktop

# Deploy
infra/k8s$> kustomize build . | kubectl apply -f -

# Delete the deployment
infra/k8s$> kustomize build . | kubectl delete -f -
```

## Connecting

**Port Forwarding**

```bash
$> kubectl port-forward mage-server 6789
```
