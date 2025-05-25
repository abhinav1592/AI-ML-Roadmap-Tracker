# AI-ML-Roadmap-Tracker
## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

* Node.js (LTS version recommended)
* npm (comes with Node.js) or yarn

### Firebase Setup

1.  **Create a Firebase Project:**
    * Go to the [Firebase Console](https://console.firebase.google.com/).
    * Click "Add project" and follow the prompts to create a new Firebase project.

2.  **Register a Web App:**
    * Once your project is created, click the web icon `</>` to add a web app to your Firebase project.
    * Follow the setup steps and copy your `firebaseConfig` object. It will look something like this:
        ```javascript
        const firebaseConfig = {
          apiKey: "YOUR_API_KEY",
          authDomain: "YOUR_AUTH_DOMAIN",
          projectId: "YOUR_PROJECT_ID",
          storageBucket: "YOUR_STORAGE_BUCKET",
          messagingSenderId: "YOUR_MESSAGING_SENDER_ID",
          appId: "YOUR_APP_ID"
        };
        ```

3.  **Enable Google Authentication:**
    * In the Firebase Console, navigate to `Authentication` in the left sidebar.
    * Go to the `Sign-in method` tab.
    * Enable the `Google` provider. Make sure to provide a public-facing app name and select a support email.

4.  **Set up Firestore Database:**
    * In the Firebase Console, navigate to `Firestore Database` in the left sidebar.
    * Click "Create database". Choose "Start in production mode" (you will set rules later).
    * Select a Cloud Firestore location (e.g., `asia-south1` for Mumbai, if you're in India).

5.  **Configure Firestore Security Rules:**
    * In Firestore Database, go to the `Rules` tab.
    * Replace the default rules with the following to allow authenticated users to read/write only their own data:
        ```firestore
        rules_version = '2';
        service cloud.firestore {
          match /databases/{database}/documents {
            // Allows read/write only if the user is authenticated and the userId matches their UID
            match /artifacts/{appId}/users/{userId}/{documents=**} {
              allow read, write: if request.auth.uid == userId;
            }
          }
        }
        ```
    * Click "Publish".

### Project Installation

1.  **Clone the repository or create using your own:**
    ```bash
    git clone [https://github.com/your-username/ai-ml-roadmap-tracker.git](https://github.com/your-username/ai-ml-roadmap-tracker.git)
    cd ai-ml-roadmap-tracker

    or
    npm create-react-app ai-ml-roadmap-tracker
    and then copy the source code from above directories
    ```
    *(Replace `https://github.com/your-username/ai-ml-roadmap-tracker.git` with the actual URL of your GitHub repository)*

2.  **Install dependencies:**
    ```bash
    npm install
    # or
    yarn install
    ```

3.  **Configure Firebase in `src/App.js`:**
    * Open `src/App.js`.
    * Locate the `firebaseConfig` object at the top of the file.
    * Replace the placeholder values (`"YOUR_API_KEY"`, etc.) with your actual Firebase configuration copied in step 2 of "Firebase Setup".

    ```javascript
    // src/App.js - Snippet
    const firebaseConfig = {
      apiKey: "YOUR_FIREBASE_API_KEY",
      authDomain: "YOUR_FIREBASE_AUTH_DOMAIN",
      projectId: "YOUR_FIREBASE_PROJECT_ID",
      storageBucket: "YOUR_FIREBASE_STORAGE_BUCKET",
      messagingSenderId: "YOUR_FIREBASE_MESSAGING_SENDER_ID",
      appId: "YOUR_FIREBASE_APP_ID"
    };
    ```

4.  **Run the application:**
    ```bash
    npm start
    # or
    yarn start
    ```
    This will open the application in your browser at `http://localhost:3000`.
