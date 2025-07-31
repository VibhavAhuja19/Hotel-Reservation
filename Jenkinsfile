pipeline {
    agent any

    environment {
        VENV_DIR = 'venv'
        GCP_PROJECT = "plated-ensign-467012-j9"
    }

    stages {
        stage('Cloning GitHub repo to Jenkins') {
            steps {
                script {
                    echo 'Cloning GitHub repo to Jenkins...'
                    git credentialsId: 'github-token', url: 'https://github.com/VibhavAhuja19/Hotel-Reservation.git', branch: 'main'
                }
            }
        }
    

        stage('Setting up Virtual Environment and Installing Dependencies') {
            steps {
                script {
                    echo 'Setting up Virtual Environment and Installing Dependencies...'
                    sh """
                    python3 -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    
                    pip install -e .
                    """
                }
            }
        }

       
    }

}