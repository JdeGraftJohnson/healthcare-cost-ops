// caj-forecast-sarima-prod1 — distributed SARIMA fan-out on ACA
//
// Each invocation gets PARTITION_ID + TOTAL_PARTITIONS via --env-vars.
// Manual trigger only; the driver fires N invocations and polls record
// blobs in stasiprod1eus2/healthcare/forecast_runs/.
//
// Pattern mirrors existing caj-fl-bench-prod1 and caj-embed-finbert-prod1.

@description('CAJ + identity environment')
param location string = resourceGroup().location
param environmentName string = 'cae-asi-prod1-eus2'   // existing ACA environment
param jobName string = 'caj-forecast-sarima-prod1'
param acrName string = 'acrasiprod1eus2'
param imageRepo string = 'asi-forecast'
param imageTag string = 'latest'

@description('Identity for storage RBAC + ACR pull')
param identityName string = 'id-caj-forecast-prod1'

@description('Storage account that hosts the silver — already provisioned')
param storageAccountName string = 'stasiprod1eus2'
param silverContainerName string = 'healthcare'

@description('Compute size per invocation. SARIMA fits sit at ~200 MB peak per series; 1.0 vCPU / 2 GiB is plenty for partition sizes up to ~2k series.')
param cpu string = '1.0'
param memory string = '2.0Gi'

@description('Job-level timeouts (seconds). Default 1 hour per partition.')
param replicaTimeoutSeconds int = 3600
param replicaRetryLimit int = 1

// ───────────────────────────────────────────────────────────────────────────
// User-assigned managed identity
// ───────────────────────────────────────────────────────────────────────────
resource caaIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: identityName
  location: location
}

// ───────────────────────────────────────────────────────────────────────────
// RBAC: Storage Blob Data Reader on the healthcare container only
// (not the whole account — financial-system workloads stay independent)
// ───────────────────────────────────────────────────────────────────────────
resource storage 'Microsoft.Storage/storageAccounts@2023-01-01' existing = {
  name: storageAccountName
}

resource container 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' existing = {
  name: '${storageAccountName}/default/${silverContainerName}'
}

resource storageReaderRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(container.id, caaIdentity.id, 'StorageBlobDataReader')
  scope: container
  properties: {
    principalId: caaIdentity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1'  // Storage Blob Data Reader
    )
  }
}

resource storageWriterRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(container.id, caaIdentity.id, 'StorageBlobDataContributor')
  scope: container
  properties: {
    principalId: caaIdentity.properties.principalId
    principalType: 'ServicePrincipal'
    // Workers write forecast_part_NNN.parquet + partition_NNN.json
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      'ba92f5b4-2d11-453d-a403-e96b0029c9fe'  // Storage Blob Data Contributor
    )
  }
}

// ───────────────────────────────────────────────────────────────────────────
// ACR pull
// ───────────────────────────────────────────────────────────────────────────
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: acrName
}

resource acrPullRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, caaIdentity.id, 'AcrPull')
  scope: acr
  properties: {
    principalId: caaIdentity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '7f951dda-4ed3-4680-a7ca-43fe172d538d'  // AcrPull
    )
  }
}

// ───────────────────────────────────────────────────────────────────────────
// Container App environment (existing)
// ───────────────────────────────────────────────────────────────────────────
resource environment 'Microsoft.App/managedEnvironments@2023-11-02-preview' existing = {
  name: environmentName
}

// ───────────────────────────────────────────────────────────────────────────
// Container App Job — Manual trigger; one invocation per partition
// ───────────────────────────────────────────────────────────────────────────
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
        parallelism: 1            // one replica per invocation; driver fires N invocations
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
          name: 'worker'
          image: '${acrName}.azurecr.io/${imageRepo}:${imageTag}'
          resources: {
            cpu: json(cpu)
            memory: memory
          }
          env: [
            // Defaults; driver overrides PARTITION_ID + TOTAL_PARTITIONS at start time.
            { name: 'AZURE_CLIENT_ID', value: caaIdentity.properties.clientId }
            { name: 'PANEL_CONFIG', value: '/app/services/forecast/configs/supplements_price.yml' }
            { name: 'PARTIAL_OUT_DIR', value: '/tmp/forecast/partial' }
            { name: 'RECORD_OUT_DIR', value: '/tmp/forecast/records' }
            { name: 'FORECAST_LOG_JSON', value: '1' }
          ]
        }
      ]
    }
  }
}

output jobName string = job.name
output identityClientId string = caaIdentity.properties.clientId
output imageReference string = '${acrName}.azurecr.io/${imageRepo}:${imageTag}'
