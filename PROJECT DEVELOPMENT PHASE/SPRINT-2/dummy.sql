SELECT * FROM user;
ALTER TABLE user add COLUMN phoneno int(10);
update user set phoneno = 1234567898 WHERE id = 2;
