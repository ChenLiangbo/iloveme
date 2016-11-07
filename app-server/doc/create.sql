use psrvdb;
create table alertapp_login(
id int(5) primary key auto_increment,
user_id int(11),
first_login datetime,
last_login datetime
is_paid int(1) default 0,
CONSTRAINT STUID FOREIGN KEY(user_id) REFERENCES auth_user(id)
)