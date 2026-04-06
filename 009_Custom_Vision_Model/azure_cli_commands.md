commandlines.

az login
az account show
az account show --query id -o tsv
az group list --output table # Shows list of resoruce group 
az group delete --name #Name of the Resource Group

az resource delete --ids 