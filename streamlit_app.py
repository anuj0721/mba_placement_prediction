import pandas as pd
import pickle
import streamlit as st


def prediction(input):
    status_prediction = status_model.predict(
        input.drop(['etest_p', 'mba_p'], axis=1))[0]

    salary_prediction = None
    if status_prediction == 1:
        salary_prediction = salary_model.predict(input)[0].round(2)

    return status_prediction, salary_prediction


def main():
    st.title('MBA Placement Prediction')

    html_temp = """
	<div style ="background-color:yellow;padding:13px">
	<h1 style ="color:black;text-align:center;">MBA Placement Prediction App </h1>
	</div>
	"""

    st.markdown(html_temp, unsafe_allow_html=True)

    gender = st.selectbox('Gender', ('M', 'F'))
    ssc_p = st.text_input(
        'Secondary Education Percentage - 10th Grade', '')
    ssc_b = st.selectbox('Secondary Education Board', ('Central', 'Others'))
    hsc_p = st.text_input(
        'Higher Secondary Education percentage - 12th Grade', '')
    hsc_b = st.selectbox('HIgher Secondary Eduction Board',
                         ('Central', 'Others'))
    hsc_s = st.selectbox(
        'Specialization in Higher Secondary Education', ('Science', 'Commerce', 'Arts'))
    degree_p = st.text_input('Degree Percentage', '')
    degree_t = st.selectbox(
        'Under Graduation(Degree type)- Field of degree education', ('Sci&Tech', 'Comm&Mgmt', 'Others'))
    workex = st.selectbox('Work Experience ', ('Yes', 'No'))
    etest_p = st.text_input('Employability test percentage', '')
    specialisation = st.selectbox(
        'Post Graduation(MBA) - Specialization', ('Mkt&Fin', 'Mkt&HR'))
    mba_p = st.text_input('MBA percentage', '')

    status = None
    salary = None
    if st.button('Predict'):
        status, salary = prediction(
            pd.DataFrame({
                'ssc_p': ssc_p,
                'hsc_p': hsc_p,
                'degree_p': degree_p,
                'etest_p': etest_p,
                'mba_p': mba_p,
                'gender': gender,
                'ssc_b': ssc_b,
                'hsc_b': hsc_b,
                'hsc_s': hsc_s,
                'degree_t': degree_t,
                'workex': workex,
                'specialisation': specialisation
            }, index=[0])
        )

    if status == None:
        pass
    elif status == 0:
        st.success('Student will not be placed.')
    else:
        st.success(f'Student will be placed with expected salary of {salary}')


if __name__ == '__main__':
    with open('status_model.pickle', 'rb') as f:
        status_model = pickle.load(f)

    with open('salary_model.pickle', 'rb') as f:
        salary_model = pickle.load(f)

    main()
