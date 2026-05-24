// caj-forecast-finalize-prod1 — single-replica downstream of the fan-out
//
// Runs after caj-forecast-sarima-prod1 has produced all forecast_part_*.parquet.
// Steps: assemble partials → ensemble → multi-α conformal on holdout →
// final-test scoring → MLflow register + champion-challenger promotion.
//
// Same UAMI + image as the fan-out worker (single image, two CMDs).

@description('CAJ + identity environment')
param location string = resourceGroup().location
param environmentName string = 'cae-asi-prod1-eus2'
param jobName string = 'caj-forecast-finalize-prod1'
param acrName string = 'acrasiprod1eus2'
param imageRepo string = 'asi-forecast'
param imageTag string = 'latest'
param identityName string = 'id-caj-forecast-prod1'   // shared with the fan-out CAJ

param storageAccountName string = 'stasiprod1eus2'
param silverContainerName string = 'healthcare'

@description('MLflow tracking URI. Empty disables registry promotion gracefully.')
param mlflowTrackingUri string = ''

@description('Compute size. Finalize touches the whole panel for ensemble + holdout; bump if SDUD-class panels OOM.')
param cpu string = '2.0'
param memory string = '4.0Gi'

param replicaTimeoutSeconds int = 1800
param replicaRetryLimit int = 1

resource caaIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = {
  name: identityName
}

resource environment 'Microsoft.App/managedEnvironments@2023-11-02-preview' existing = {
  name: environmentName
}

resource job 'Microsoft.App/jobs@2023-11-02-preview' = {
  name: jobName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${caaIdentity.id}': {}
    }
  }
  properties: {
    environmentId: environment.id
    configuration: {
      triggerType: 'Manual'
      replicaTimeout: replicaTimeoutSeconds
      replicaRetryLimit: replicaRetryLimit
      manualTriggerConfig: {
        parallelism: 1
        replicaCompletionCount: 1
      }
      registries: [
        {
          server: '${acrName}.azurecr.io'
          identity: caaIdentity.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'finalize'
          image: '${acrName}.azurecr.io/${imageRepo}:${imageTag}'
          resources: {
            cpu: json(cpu)
            memory: memory
          }
          // Override CMD to call finalize instead of the worker.
          command: [
            '/usr/bin/tini'
            '--'
            'python'
            '-m'
            'services.forecast.finalize'
          ]
          args: [
            '--config'
            '/app/services/forecast/configs/supplements_price.yml'
            '--partial-dir'
            '/mnt/silver/forecast_runs/partial'
            '--out'
            '/mnt/silver/silver/snapshot=$(date +%Y-%m-%d)/forecast.parquet'
            '--register'
            '--bundle-path'
            '/mnt/silver/forecast_runs/bundle'
            '--champion-tol'
            '0.05'
            '--verbose'
          ]
          env: [
            { name: 'AZURE_CLIENT_ID', value: caaIdentity.properties.clientId }
            { name: 'FORECAST_LOG_JSON', value: '1' }
            { name: 'MLFLOW_TRACKING_URI', value: mlflowTrackingUri }
            { name: 'MLFLOW_EXPERIMENT', value: 'forecast' }
          ]
        }
      ]
    }
  }
}

output jobName string = job.name
output imageReference string = '${acrName}.azurecr.io/${imageRepo}:${imageTag}'
