import streamlit as st;
import bcrypt;
from userdb import userList;

def hashPassword(password): #only including to show process, would not be in production
    return bcrypt.hashpw(password.encode('utf-8'),bcrypt.gensalt())

def verify(name, password) -> bool: 
    if (name in userList):
        if (bcrypt.checkpw(password.encode('utf-8'),userList[name])):
            return True
        else:
            st.error("Incorrect Password")
    return False

def login():
    st.title("YOLO Login")
    username = st.text_input("Username")
    password = st.text_input('Password',type='password')

    if (st.button("Login")):
        if verify(username, password):
            st.success(f"Welcome {username} !")
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.switch_page("pages/app.py")
        else:
            st.error("Incorrect Username")
if "authenticated" not in st.session_state:
    login()
else:
    st.write(f"Currently logged in as {st.session_state['username']}")
    if (st.button("Logout")):
        st.session_state.clear()