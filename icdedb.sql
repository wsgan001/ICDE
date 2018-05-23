DROP DATABASE IF EXISTS icdedb;
CREATE DATABASE icdedb;
USE icdedb;

CREATE TABLE cards(
id INT(10) not null auto_increment,
lname VARCHAR(50) not null,
fname VARCHAR(50) not null,
mname VARCHAR(50),
id_type VARCHAR(50),
validity DATE,
constraint cards_id_pk primary key(id) 
);