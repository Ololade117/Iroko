-- ==========================================================
-- AI HOSPITAL DATABASE
-- PostgreSQL 16+
-- ==========================================================

CREATE DATABASE ai_hospital;
-- Connect to ai_hospital before running the statements below.
-- \c ai_hospital

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- =========================
-- ROLES
-- =========================
CREATE TABLE roles(
 id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
 name VARCHAR(50) UNIQUE NOT NULL
);

INSERT INTO roles(name) VALUES
('ADMIN'),('DOCTOR'),('NURSE'),('PATIENT');

-- =========================
-- USERS
-- =========================
CREATE TABLE users(
 id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
 role_id UUID REFERENCES roles(id),
 first_name VARCHAR(100) NOT NULL,
 last_name VARCHAR(100) NOT NULL,
 email VARCHAR(255) UNIQUE NOT NULL,
 password_hash TEXT NOT NULL,
 phone VARCHAR(20),
 preferred_language VARCHAR(20) DEFAULT 'en',
 is_active BOOLEAN DEFAULT TRUE,
 created_at TIMESTAMP DEFAULT NOW()
);

-- =========================
-- DEPARTMENTS
-- =========================
CREATE TABLE departments(
 id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
 name VARCHAR(100) UNIQUE NOT NULL
);

INSERT INTO departments(name) VALUES
('Emergency'),
('Cardiology'),
('General Medicine'),
('Paediatrics');

-- =========================
-- DOCTORS
-- =========================
CREATE TABLE doctors(
 id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
 user_id UUID UNIQUE REFERENCES users(id) ON DELETE CASCADE,
 department_id UUID REFERENCES departments(id),
 licence_no VARCHAR(100) UNIQUE,
 speciality VARCHAR(100)
);

-- =========================
-- PATIENTS
-- =========================
CREATE TABLE patients(
 id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
 user_id UUID UNIQUE REFERENCES users(id) ON DELETE CASCADE,
 gender VARCHAR(20),
 date_of_birth DATE,
 blood_group VARCHAR(5),
 height_cm NUMERIC(5,2),
 weight_kg NUMERIC(5,2),
 emergency_contact_name VARCHAR(120),
 emergency_contact_phone VARCHAR(20)
);

-- =========================
-- MEDICAL HISTORY
-- =========================
CREATE TABLE medical_history(
 id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
 patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
 illness VARCHAR(200) NOT NULL,
 date_contracted DATE,
 duration_days INT,
 symptoms TEXT,
 severity VARCHAR(20),
 diagnosis TEXT,
 treatment TEXT,
 medications TEXT,
 recovery_status VARCHAR(30),
 allergies TEXT,
 chronic_conditions TEXT,
 doctor_notes TEXT,
 recorded_at TIMESTAMP DEFAULT NOW()
);

-- =========================
-- APPOINTMENTS
-- =========================
CREATE TABLE appointments(
 id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
 patient_id UUID REFERENCES patients(id),
 doctor_id UUID REFERENCES doctors(id),
 appointment_date TIMESTAMP,
 reason TEXT,
 status VARCHAR(30) DEFAULT 'Scheduled'
);

-- =========================
-- VITAL SIGNS
-- =========================
CREATE TABLE vital_signs(
 id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
 patient_id UUID REFERENCES patients(id),
 temperature NUMERIC(4,1),
 heart_rate INT,
 blood_pressure VARCHAR(20),
 oxygen_saturation INT,
 weight_kg NUMERIC(5,2),
 recorded_at TIMESTAMP DEFAULT NOW()
);

-- =========================
-- LAB RESULTS
-- =========================
CREATE TABLE lab_results(
 id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
 patient_id UUID REFERENCES patients(id),
 test_name VARCHAR(150),
 result TEXT,
 status VARCHAR(30),
 test_date DATE
);

-- =========================
-- AI CONSULTATIONS
-- =========================
CREATE TABLE ai_consultations(
 id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
 patient_id UUID REFERENCES patients(id),
 original_query TEXT,
 translated_query TEXT,
 triage_level VARCHAR(20),
 probable_diagnosis TEXT,
 recommendation TEXT,
 created_at TIMESTAMP DEFAULT NOW()
);

-- =========================
-- RAG DOCUMENTS
-- =========================
CREATE TABLE rag_documents(
 id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
 title VARCHAR(255),
 category VARCHAR(100),
 content TEXT,
 source VARCHAR(255)
);

-- =========================
-- AUDIT LOGS
-- =========================
CREATE TABLE audit_logs(
 id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
 user_id UUID REFERENCES users(id),
 action TEXT,
 created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_history_patient ON medical_history(patient_id);
CREATE INDEX idx_consult_patient ON ai_consultations(patient_id);
CREATE INDEX idx_lab_patient ON lab_results(patient_id);

-- =========================
-- DUMMY DATA
-- =========================
INSERT INTO users(role_id,first_name,last_name,email,password_hash,phone)
SELECT id,'Grace','Admin','admin@hospital.test',
'$2y$12$dummybcrypthash','08030000001'
FROM roles WHERE name='ADMIN';

INSERT INTO users(role_id,first_name,last_name,email,password_hash,phone)
SELECT id,'James','Wilson','doctor@hospital.test',
'$2y$12$dummybcrypthash','08030000002'
FROM roles WHERE name='DOCTOR';

INSERT INTO users(role_id,first_name,last_name,email,password_hash,phone,preferred_language)
SELECT id,'John','Doe','patient@hospital.test',
'$2y$12$dummybcrypthash','08030000003','en'
FROM roles WHERE name='PATIENT';

INSERT INTO doctors(user_id,department_id,licence_no,speciality)
SELECT u.id,d.id,'DOC-1001','General Medicine'
FROM users u, departments d
WHERE u.email='doctor@hospital.test' AND d.name='General Medicine';

INSERT INTO patients(user_id,gender,date_of_birth,blood_group,height_cm,weight_kg,emergency_contact_name,emergency_contact_phone)
SELECT id,'Male','1992-05-12','O+',178,78,'Jane Doe','08031111111'
FROM users WHERE email='patient@hospital.test';

INSERT INTO medical_history(
patient_id,illness,date_contracted,duration_days,symptoms,severity,diagnosis,
treatment,medications,recovery_status,allergies,chronic_conditions,doctor_notes)
SELECT id,
'Malaria',
'2026-07-20',
5,
'High fever, chills, headache, fatigue',
'Moderate',
'Confirmed malaria',
'Artemisinin combination therapy',
'Artemether/Lumefantrine',
'Recovered',
'Penicillin',
'Hypertension',
'Patient responded well'
FROM patients;

INSERT INTO appointments(patient_id,doctor_id,appointment_date,reason,status)
SELECT p.id,d.id,NOW() + interval '2 days','Follow-up review','Scheduled'
FROM patients p CROSS JOIN doctors d LIMIT 1;

INSERT INTO vital_signs(patient_id,temperature,heart_rate,blood_pressure,oxygen_saturation,weight_kg)
SELECT id,38.6,92,'130/85',98,78 FROM patients;

INSERT INTO lab_results(patient_id,test_name,result,status,test_date)
SELECT id,'Malaria Rapid Test','Positive','Completed',CURRENT_DATE FROM patients;

INSERT INTO ai_consultations(
patient_id,original_query,translated_query,triage_level,
probable_diagnosis,recommendation)
SELECT id,
'I have had fever and chills for five days',
'I have had fever and chills for five days',
'MEDIUM',
'Possible malaria',
'Visit the nearest hospital for a malaria test and begin treatment if confirmed.'
FROM patients;

INSERT INTO rag_documents(title,category,content,source)
VALUES
('WHO Malaria Guideline','Malaria',
'Malaria diagnosis should be confirmed using a parasitological test before treatment where possible.',
'WHO'),
('Hypertension Overview','Cardiology',
'Persistent blood pressure >=140/90 mmHg requires clinical evaluation.',
'WHO');

INSERT INTO audit_logs(user_id,action)
SELECT id,'Initial system seed'
FROM users;

-- Verification
SELECT COUNT(*) AS users FROM users;
SELECT COUNT(*) AS patients FROM patients;
SELECT COUNT(*) AS history FROM medical_history;
SELECT COUNT(*) AS consultations FROM ai_consultations;
