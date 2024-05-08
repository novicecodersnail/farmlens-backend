# Azure Function for Model Query

This Azure Function queries a machine learning model and returns the results to the client. It is designed to integrate seamlessly with Azure's cloud infrastructure and requires setup with Azure CLI and appropriate environment settings.

## Prerequisites

Before you begin, ensure you have the following installed:
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
- Python 3.7 or later

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Setting up Azure CLI

Ensure that Azure CLI is installed on your machine. Azure CLI allows you to manage Azure resources directly from the command line. For installation instructions, visit [Install Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli).

### Cloning the Repository

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/yourusername/your-repository-name.git
cd your-repository-name

Setting up the Virtual Environment
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate

Installing Dependencies
Install all the required dependencies by running:
pip install -r requirements.txt

Local Settings
You must configure your local settings according to the Azure Function requirements. This typically includes setting up local environment variables for development:
cp local.settings.json.example local.settings.json
<Alternatively as these setting are hidden, you should go to Azure portal and get them from enviroment variables >

Running Locally
To run the Azure Function locally, use the following command:
func start

Deployment
To deploy this function to Azure:

Log in to Azure using the Azure CLI:
az login

Deploy the function using the Azure Function Core Tools:
func azure functionapp publish farmlensfunctionapp
