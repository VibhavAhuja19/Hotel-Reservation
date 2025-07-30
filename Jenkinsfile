pipeline {
    agent any

    environment {
        VENV_DIR = 'venv'
        GCP_PROJECT = "plated-ensign-467012-j9"
    }

    stages {

        stage('Clone GitHub Repo') {
            steps {
                script {
                    echo '📥 Cloning GitHub repo into Jenkins...'
                    checkout scmGit(
                        branches: [[name: '*/main']],
                        extensions: [],
                        userRemoteConfigs: [[
                            credentialsId: 'github-token',
                            url: 'https://github.com/VibhavAhuja19/Hotel-Reservation.git'
                        ]]
                    )
                }
            }
        }

        stage('Set Up Virtual Environment and Install Dependencies') {
            steps {
                script {
                    echo '🐍 Setting up virtual environment and installing dependencies...'
                    sh '''
                        python3 -m venv ${VENV_DIR}
                        . ${VENV_DIR}/bin/activate
                        pip install --upgrade pip
                        pip install -e .
                    '''
                }
            }
        }

        stage('Install gcloud CLI (if not already installed)') {
            steps {
                script {
                    echo '☁️ Installing gcloud CLI...'
                    sh '''
                        if ! command -v gcloud &> /dev/null
                        then
                            echo "gcloud CLI not found, installing..."
                            apt-get update -y
                            apt-get install -y curl apt-transport-https ca-certificates gnupg
                            curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
                            echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
                            apt-get update && apt-get install -y google-cloud-sdk
                        else
                            echo "gcloud CLI already installed."
                        fi
                    '''
                }
            }
        }

        stage('Build & Push Docker Image to GCR') {
            steps {
                withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                    script {
                        echo '🐳 Building and pushing Docker image to GCR...'
                        sh '''
                            gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                            gcloud config set project ${GCP_PROJECT}
                            gcloud auth configure-docker --quiet

                            docker build -t gcr.io/${GCP_PROJECT}/ml-project:latest .
                            docker push gcr.io/${GCP_PROJECT}/ml-project:latest
                        '''
                    }
                }
            }
        }
    }
}
