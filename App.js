import React, { useState, useEffect, useCallback, createContext, useContext } from 'react';
import { initializeApp } from 'firebase/app';
import { getAuth, onAuthStateChanged, GoogleAuthProvider, signInWithPopup, signOut } from 'firebase/auth'; // Added signOut, removed signInAnonymously, signInWithCustomToken
import { getFirestore, doc, setDoc, onSnapshot } from 'firebase/firestore';
import './App.css'; // Ensure this file is located in the 'src/' directory alongside App.js
import roadmapData from './AiRoadMapData'; // Import roadmapData from the new file


// Define firebaseConfig directly for local development.
// REPLACE THE FOLLOWING WITH YOUR ACTUAL FIREBASE CONFIG FROM YOUR PROJECT SETTINGS!
const firebaseConfig = {
  apiKey: "YOUR_API_KEY", // <--- REPLACE WITH YOUR API KEY
  authDomain: "YOUR_AUTH_DOMAIN", // <--- REPLACE WITH YOUR AUTH DOMAIN
  projectId: "YOUR_PROJECT_ID", // <--- REPLACE WITH YOUR PROJECT ID
  storageBucket: "YOUR_STORAGE_BUCKET", // <--- REPLACE WITH YOUR STORAGE BUCKET
  messagingSenderId: "YOUR_MESSAGING_SENDER_ID", // <--- REPLACE WITH YOUR MESSAGING SENDER ID
  appId: "YOUR_APP_ID" // <--- REPLACE WITH YOUR APP ID
};


// Array of congratulatory messages
const congratulatoryMessages = [
  "Fantastic! You've crushed this phase and are one step closer to AI mastery!",
  "Incredible work! Your dedication is truly paying off. Keep building!",
  "Phase complete! You're making remarkable progress on your journey to AI leadership. Onward!",
  "Boom! Another phase conquered! Your commitment is inspiring. What's next?",
  "Congratulations! You've successfully navigated this challenge. Celebrate your win!",
  "Excellent job! Each step brings you closer to your AI goals. Keep that momentum going!",
  "You did it! This milestone is a testament to your hard work. Time to level up!"
];

// Firebase context to provide db, auth, userId, and signInWithGoogle to components
const FirebaseContext = createContext(null);

// Custom hook for Firebase initialization and auth
const useFirebase = () => {
  const [db, setDb] = useState(null);
  const [auth, setAuth] = useState(null);
  const [userId, setUserId] = useState(null);
  const [userEmail, setUserEmail] = useState(null); // New state for user email
  const [appId, setAppId] = useState(null); // State to hold appId
  const [isAuthReady, setIsAuthReady] = useState(false);

  useEffect(() => {
    try {
      // Determine appId: prioritize Canvas global if available, otherwise use localFirebaseConfig or a default
      const currentAppId = typeof window !== 'undefined' && typeof window.__app_id !== 'undefined' ? window.__app_id : firebaseConfig.appId || 'default-local-app-id';
      setAppId(currentAppId);

      // Initialize Firebase app using the determined config
      const app = initializeApp(firebaseConfig);
      const firestore = getFirestore(app);
      const firebaseAuth = getAuth(app);

      setDb(firestore);
      setAuth(firebaseAuth);

      // FOR DEVELOPMENT/TESTING: Force sign out on component mount to always show login page
      // In a production app, you might only sign out on explicit user action.
      signOut(firebaseAuth).then(() => {
        console.log("Forced sign-out on load for development/testing.");
      }).catch((error) => {
        console.error("Error during forced sign-out:", error);
      });

      // Listen for auth state changes
      const unsubscribe = onAuthStateChanged(firebaseAuth, (user) => {
        if (user) {
          setUserId(user.uid);
          setUserEmail(user.email); // Set user email
          console.log("onAuthStateChanged: User is authenticated with UID:", user.uid, "Email:", user.email);
        } else {
          setUserId(null); // Explicitly set to null if not authenticated
          setUserEmail(null); // Clear user email
          console.log("onAuthStateChanged: User is not authenticated (user object is null).");
        }
        setIsAuthReady(true); // Auth state is now known.
        console.log("onAuthStateChanged: Auth state is ready.");
      });

      return () => unsubscribe();
    } catch (error) {
      console.error("Failed to initialize Firebase:", error);
    }
  }, []);

  const signInWithGoogle = useCallback(async () => {
    if (!auth) {
      console.error("Auth object not initialized.");
      return;
    }
    const provider = new GoogleAuthProvider();
    try {
      console.log("Attempting Google Sign-In popup...");
      await signInWithPopup(auth, provider);
      console.log("Google Sign-In successful.");
    } catch (error) {
      console.error("Error during Google Sign-In:", error);
    }
  }, [auth]);

  const handleSignOut = useCallback(async () => {
    if (!auth) {
      console.error("Auth object not initialized for sign out.");
      return;
    }
    try {
      await signOut(auth);
      console.log("User signed out successfully.");
    } catch (error) {
      console.error("Error signing out:", error);
    }
  }, [auth]);

  return { db, auth, userId, userEmail, appId, isAuthReady, signInWithGoogle, handleSignOut };
};

// ProgressBar Component
const ProgressBar = ({ progress }) => {
  return (
    <div className="progress-bar-container">
      <div
        className="progress-bar-fill"
        style={{ width: `${progress}%` }}
      ></div>
    </div>
  );
};

// Modal Component for messages
const Modal = ({ message, onClose }) => {
  if (!message) return null;
  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <p className="modal-message">{message}</p>
        <button
          onClick={onClose}
          className="modal-button"
        >
          Got It!
        </button>
      </div>
    </div>
  );
};

// DailyReminder Component
const DailyReminder = () => {
  const { userId, db, appId } = useContext(FirebaseContext); // Get appId from context
  const [showReminder, setShowReminder] = useState(false);
  const roadmapId = "ai-ml-career-roadmap"; // Fixed ID for this roadmap

  useEffect(() => {
    if (!db || !userId || !appId) {
      console.log("DailyReminder: Skipping fetch. db, userId, or appId is null/undefined.", { db, userId, appId });
      return; // Ensure appId and userId are available and not null
    }

    const reminderDocRef = doc(db, `artifacts/${appId}/users/${userId}/reminders`, roadmapId);
    console.log(`DailyReminder: Attempting to fetch from path: artifacts/${appId}/users/${userId}/reminders/${roadmapId}`);

    const unsubscribe = onSnapshot(reminderDocRef, (docSnap) => {
      if (docSnap.exists()) {
        const data = docSnap.data();
        const storedDate = data.lastReminderDate;

        const today = new Date().toDateString();
        if (storedDate !== today) {
          setShowReminder(true);
        }
      } else {
        // If no reminder doc exists, show reminder for the first time
        setShowReminder(true);
      }
    }, (error) => {
      console.error("Error fetching daily reminder:", error);
    });

    return () => unsubscribe();
  }, [db, userId, appId]); // Add appId to dependencies

  const handleCloseReminder = async () => {
    setShowReminder(false);
    if (db && userId && appId) { // Ensure appId and userId are available and not null
      const today = new Date().toDateString();
      const reminderDocRef = doc(db, `artifacts/${appId}/users/${userId}/reminders`, roadmapId);
      try {
        await setDoc(reminderDocRef, { lastReminderDate: today }, { merge: true });
      } catch (error) {
        console.error("Error updating last reminder date:", error);
      }
    } else {
      console.warn("DailyReminder: Cannot close reminder, Firebase not ready or user not authenticated.");
    }
  };

  return (
    <Modal
      message={showReminder ? "Don't forget to dedicate 30-60 minutes to your AI/ML roadmap today!" : null}
      onClose={handleCloseReminder}
    />
  );
};

// Individual Daily Task / Project Item Component
const DailyTaskItem = ({ item, onUpdateItem, userProgress }) => {
  const itemId = item.id; // This is the unique ID for this specific daily item
  const [completed, setCompleted] = useState(userProgress?.[itemId]?.completed || false);
  const [workLink, setWorkLink] = useState(userProgress?.[itemId]?.workLink || '');
  const [notes, setNotes] = useState(userProgress?.[itemId]?.notes || '');

  useEffect(() => {
    setCompleted(userProgress?.[itemId]?.completed || false);
    setWorkLink(userProgress?.[itemId]?.workLink || '');
    setNotes(userProgress?.[itemId]?.notes || '');
  }, [userProgress, itemId]);

  const handleChange = useCallback((field, value) => {
    const updatedState = {
      completed: field === 'completed' ? value : completed,
      workLink: field === 'workLink' ? value : workLink,
      notes: field === 'notes' ? value : notes,
    };
    if (field === 'completed') setCompleted(value);
    if (field === 'workLink') setWorkLink(value);
    if (field === 'notes') setNotes(value);

    // Defensive check before calling onUpdateItem
    if (typeof onUpdateItem === 'function') {
      onUpdateItem(itemId, updatedState);
    } else {
      console.error("DailyTaskItem: onUpdateItem is not a function!", onUpdateItem);
    }
  }, [completed, workLink, notes, onUpdateItem, itemId]); // onUpdateItem is in dependencies

  // Log the type of onUpdateItem when the component renders or updates
  // useEffect(() => {
  //   console.log(`DailyTaskItem (${itemId}): typeof onUpdateItem = ${typeof onUpdateItem}`);
  // }, [onUpdateItem, itemId]);


  return (
    <div className="daily-task-item">
      <div className="daily-task-main-content">
        <input
          type="checkbox"
          checked={completed}
          onChange={(e) => handleChange('completed', e.target.checked)}
          className="task-checkbox"
        />
        <span className={`task-description ${completed ? 'completed-task' : ''}`}>
          {item.description}
        </span>
        {item.resources && item.resources.length > 0 && (
          <div className="task-resources">
            {item.resources.map((res, idx) => (
              <a
                key={idx}
                href={res.url}
                target="_blank"
                rel="noopener noreferrer"
                className="resource-link"
              >
                {res.name}
              </a>
            ))}
          </div>
        )}
      </div>
      <div className="task-inputs">
        <input
          type="text"
          placeholder="Link to work/notes"
          value={workLink}
          onChange={(e) => handleChange('workLink', e.target.value)}
          className="task-input"
        />
        <input
          type="text"
          placeholder="Add quick notes"
          value={notes}
          onChange={(e) => handleChange('notes', e.target.value)}
          className="task-input"
        />
      </div>
    </div>
  );
};

// Week Component
const Week = ({ week, onUpdateTask, userProgress }) => {
  // Calculate if all daily tasks/projects in this week are completed
  const isWeekCompleted = week.dailyTasks.every(item => userProgress?.[item.id]?.completed);

  // Log the type of onUpdateTask when the component renders or updates
  // useEffect(() => {
  //   console.log(`Week (${week.id}): typeof onUpdateTask = ${typeof onUpdateTask}`);
  // }, [onUpdateTask, week.id]);

  return (
    <div className="week-container">
      <div className="week-header">
        <input
          type="checkbox"
          checked={isWeekCompleted}
          readOnly // Make it read-only as it's derived
          className="week-checkbox"
        />
        <h4 className="week-title">{week.title}</h4>
      </div>
      {week.dailyTasks.map(item => (
        <DailyTaskItem
          key={item.id}
          item={item}
          onUpdateItem={onUpdateTask}
          userProgress={userProgress}
        />
      ))}
    </div>
  );
};

// Phase Component
const Phase = ({ phase, onUpdateTask, userProgress, phaseProgress, onPhaseComplete }) => {
  const [showCongratulation, setShowCongratulation] = useState(false);
  const [hasCelebrated, setHasCelebrated] = useState(false);
  const [currentCongratsMessage, setCurrentCongratsMessage] = useState('');

  useEffect(() => {
    if (phaseProgress === 100 && !hasCelebrated) {
      const randomIndex = Math.floor(Math.random() * congratulatoryMessages.length);
      setCurrentCongratsMessage(congratulatoryMessages[randomIndex]);
      setShowCongratulation(true);
      setHasCelebrated(true); // Prevent showing multiple times
      onPhaseComplete(phase.id); // Notify parent of completion
    }
  }, [phaseProgress, hasCelebrated, phase.id, onPhaseComplete]);

  const handleCloseCongratulation = () => {
    setShowCongratulation(false);
  };

  // Log the type of onUpdateTask when the component renders or updates
  // useEffect(() => {
  //   console.log(`Phase (${phase.id}): typeof onUpdateTask = ${typeof onUpdateTask}`);
  // }, [onUpdateTask, phase.id]);

  return (
    <div className="phase-container">
      <h3 className="phase-title">{phase.title}</h3>
      <div className="phase-progress-section">
        <p className="progress-text">Phase Progress: {phaseProgress.toFixed(0)}%</p>
        <ProgressBar progress={phaseProgress} />
      </div>
      {phase.weeks.map(week => (
        <Week
          key={week.id}
          week={week}
          onUpdateTask={onUpdateTask}
          userProgress={userProgress}
        />
      ))}
      <Modal
        message={showCongratulation ? currentCongratsMessage : null}
        onClose={handleCloseCongratulation}
      />
    </div>
  );
};

// Main RoadmapTracker Component
const RoadmapTracker = () => {
  const { userId, userEmail, db, appId, isAuthReady, handleSignOut } = useContext(FirebaseContext); // Get handleSignOut from context
  const [userProgress, setUserProgress] = useState({});
  const [overallProgress, setOverallProgress] = useState(0);
  const [phaseProgresses, setPhaseProgresses] = useState({});
  const [postPhaseNotes, setPostPhaseNotes] = useState('');

  const roadmapId = "ai-ml-career-roadmap"; // Fixed ID for this roadmap

  // Fetch user progress from Firestore
  useEffect(() => {
    if (!db || !userId || !appId || !isAuthReady) {
      console.log("RoadmapTracker: Skipping fetch. db, userId, appId, or isAuthReady is null/undefined.", { db, userId, appId, isAuthReady });
      return; // Ensure appId and userId are available and not null
    }

    const userRoadmapDocRef = doc(db, `artifacts/${appId}/users/${userId}/roadmaps`, roadmapId);
    console.log(`RoadmapTracker: Attempting to fetch from path: artifacts/${appId}/users/${userId}/roadmaps/${roadmapId}`);

    const unsubscribe = onSnapshot(userRoadmapDocRef, (docSnap) => {
      if (docSnap.exists()) {
        const data = docSnap.data();
        setUserProgress(data.tasks || {});
        setPostPhaseNotes(data.postPhaseNotes || '');
        console.log("RoadmapTracker: User progress data loaded successfully.");
      } else {
        // Initialize if document doesn't exist
        console.log("RoadmapTracker: No user roadmap data found, initializing empty progress.");
        setUserProgress({});
        setPostPhaseNotes('');
      }
    }, (error) => {
      console.error("RoadmapTracker: Error fetching user progress:", error);
    });

    return () => unsubscribe();
  }, [db, userId, appId, isAuthReady]); // Add appId to dependencies

  // Calculate progress whenever userProgress changes
  useEffect(() => {
    if (!userProgress) return;

    let totalItems = 0;
    let completedItems = 0;
    const newPhaseProgresses = {};

    roadmapData.forEach(phase => {
      let phaseTotalItems = 0;
      let phaseCompletedItems = 0;

      phase.weeks.forEach(week => {
        week.dailyTasks.forEach(item => { // Iterate over dailyTasks directly
          totalItems++;
          phaseTotalItems++;
          if (userProgress[item.id]?.completed) {
            completedItems++;
            phaseCompletedItems++;
          }
        });
      });
      newPhaseProgresses[phase.id] = phaseTotalItems > 0 ? (phaseCompletedItems / phaseTotalItems) * 100 : 0;
    });

    setPhaseProgresses(newPhaseProgresses);
    setOverallProgress(totalItems > 0 ? (completedItems / totalItems) * 100 : 0);
  }, [userProgress]);

  // Function to update a task's status in Firestore
  const handleUpdateTask = useCallback(async (itemId, updates) => {
    if (!db || !userId || !appId) { // Ensure appId and userId are available and not null
      console.warn("Firestore not ready or user not authenticated. Cannot update task.");
      return;
    }
    const userRoadmapDocRef = doc(db, `artifacts/${appId}/users/${userId}/roadmaps`, roadmapId);
    try {
      // Create a copy to avoid direct mutation of state
      const updatedUserProgress = { ...userProgress, [itemId]: { ...userProgress[itemId], ...updates } };
      setUserProgress(updatedUserProgress); // Optimistic UI update
      console.log(`Attempting to save task ${itemId} to Firestore for user ${userId}.`);
      await setDoc(userRoadmapDocRef, { tasks: updatedUserProgress }, { merge: true });
      console.log(`Task ${itemId} saved successfully.`);
    } catch (error) {
      console.error("Error updating task:", error);
      // Revert optimistic update if error occurs
      setUserProgress(userProgress);
    }
  }, [db, userId, appId, userProgress]); // Add appId to dependencies

  // Function to update post-phase notes
  const handleUpdatePostPhaseNotes = useCallback(async (notes) => {
    if (!db || !userId || !appId) { // Ensure appId and userId are available and not null
      console.warn("Firestore not ready or user not authenticated. Cannot update post-phase notes.");
      return;
    }
    const userRoadmapDocRef = doc(db, `artifacts/${appId}/users/${userId}/roadmaps`, roadmapId);
    try {
      setPostPhaseNotes(notes); // Optimistic UI update
      console.log(`Attempting to save post-phase notes to Firestore for user ${userId}.`);
      await setDoc(userRoadmapDocRef, { postPhaseNotes: notes }, { merge: true });
      console.log("Post-phase notes saved successfully.");
    } catch (error) {
      console.error("Error updating post-phase notes:", error);
      setPostPhaseNotes(postPhaseNotes); // Revert optimistic update
    }
  }, [db, userId, appId, postPhaseNotes]); // Add appId to dependencies

  const handlePhaseComplete = useCallback((phaseId) => {
    // This function can be used to trigger specific actions or messages
    // when a phase is completed, beyond the modal in the Phase component itself.
    console.log(`Phase ${phaseId} completed!`);
  }, []);

  // Log the type of handleUpdateTask when the component renders or updates
  // useEffect(() => {
  //   console.log(`RoadmapTracker: typeof handleUpdateTask = ${typeof handleUpdateTask}`);
  // }, [handleUpdateTask]);

  return (
    <div className="app-container">
      <h1 className="main-heading">AI/ML Career Roadmap Tracker</h1>

      {userId && (
        <div className="user-id-display">
          Your User ID: <span className="user-id-value">{userEmail || userId}</span>
          <button onClick={handleSignOut} className="sign-out-button">Sign Out</button>
        </div>
      )}

      <div className="overall-progress-card">
        <h2 className="card-title">Overall Progress</h2>
        <p className="progress-text">Total Progress: {overallProgress.toFixed(0)}%</p>
        <ProgressBar progress={overallProgress} />
      </div>

      {roadmapData.map(phase => (
        <Phase
          key={phase.id}
          phase={phase}
          onUpdateTask={handleUpdateTask}
          userProgress={userProgress}
          phaseProgress={phaseProgresses[phase.id] || 0}
          onPhaseComplete={handlePhaseComplete}
        />
      ))}

      <div className="post-phases-card">
        <h2 className="card-title">Work Done Post-Phases</h2>
        <textarea
          className="post-phases-textarea"
          placeholder="Add notes, projects, or links for work done beyond the defined phases..."
          value={postPhaseNotes}
          onChange={(e) => handleUpdatePostPhaseNotes(e.target.value)}
        ></textarea>
      </div>

      {/* Future Roadmap Creation Placeholder */}
      <div className="new-roadmap-card">
        <h2 className="card-title">Add New Learning Roadmaps (Future Feature)</h2>
        <p className="new-roadmap-text">
          In the future, you'll be able to add new learning roadmaps here and track your progress for various topics.
        </p>
        <button
          className="new-roadmap-button"
          disabled // Disable for now as it's a future feature
        >
          Create New Roadmap
        </button>
      </div>
      <DailyReminder />
    </div>
  );
};

// LoginPage Component
const LoginPage = () => {
  const { signInWithGoogle } = useContext(FirebaseContext);

  return (
    <div className="login-page-container">
      <h1 className="login-heading">Welcome to AI/ML Career Roadmap</h1>
      <p className="login-subheading">Please sign in to track your progress.</p>
      <button
        onClick={signInWithGoogle}
        className="google-sign-in-button"
      >
        <img
          src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg"
          alt="Google logo"
          className="google-logo"
        />
        Sign in with Google
      </button>
    </div>
  );
};

// App Wrapper with Firebase Context Provider
const App = () => {
  const { db, auth, userId, appId, isAuthReady, signInWithGoogle, handleSignOut } = useFirebase();

  if (!isAuthReady) {
    return (
      <div className="loading-container">
        <div className="loading-text">Loading tracker...</div>
      </div>
    );
  }

  return (
    <FirebaseContext.Provider value={{ db, auth, userId, appId, isAuthReady, signInWithGoogle, handleSignOut }}>
      {userId ? <RoadmapTracker /> : <LoginPage />}
    </FirebaseContext.Provider>
  );
};

export default App;
