pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                echo sh(returnStdout: true, script: 'env')

                slackSend "Build Started - ${env.JOB_NAME} ${env.BUILD_NUMBER} ${env.BRANCH_NAME} (<${env.BUILD_URL}|Open>)"

                sh '''
                    conda create --yes -n ${BUILD_TAG} python
                    source activate ${BUILD_TAG}

                    pip install --only-binary=numpy,scipy,scikit-learn,astropy numpy scipy scikit-learn astropy
                    pip install healpy
                    pip install nose
                    python setup.py install
                '''
            }
        }
        stage('Test') {
            steps {
                sh '''
                    source activate ${BUILD_TAG}
                    nosetests
                '''
            }
        }
    }
    post {
        always {
            sh '''
                source deactivate ${BUILD_TAG}
                conda remove --yes -n ${BUILD_TAG} --all
            '''
        }
        success {
            slackSend "Passed! - ${env.JOB_NAME} ${env.BUILD_NUMBER} ${env.BRANCH_NAME} (<${env.BUILD_URL}|Open>)"
        }
        unstable {
            slackSend "Unstable! - ${env.JOB_NAME} ${env.BUILD_NUMBER} ${env.BRANCH_NAME} (<${env.BUILD_URL}|Open>)"
        }
        failure {
            slackSend "Failed! - ${env.JOB_NAME} ${env.BUILD_NUMBER} ${env.BRANCH_NAME} (<${env.BUILD_URL}|Open>)"
        }
        changed {
            slackSend "Changed! - ${env.JOB_NAME} ${env.BUILD_NUMBER} ${env.BRANCH_NAME} (<${env.BUILD_URL}|Open>)"
        }

    }
}
