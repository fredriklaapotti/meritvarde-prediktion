drop table if exists sg;
drop table if exists clean;

create table sg (
    sourcename varchar(255),
    school varchar(255),
    year varchar(32),
    class varchar(32),
    semester varchar(255),
    schoolyear varchar(255),
    subject varchar(255),
    cohort varchar(32),
    grade char(10),
    points decimal(3,1) unsigned,
    reachedgoals varchar(32),
    sex varchar(32),
    mothertongue varchar(255),
    created varchar(32),
    updated varchar(32),
    ssn varchar(32),
    status varchar(255),
    transferredgrade varchar(32)
);

create table clean (
    ssn char(10),
    sex varchar(16),
    mothertongue varchar(64),
    -- Svenska
    g6_sv decimal(3,1) unsigned,
    g7_sv decimal(3,1) unsigned,
    g8_sv decimal(3,1) unsigned,
    g9_sv decimal(3,1) unsigned,
    -- Svenska som andraspråk
    g6_sva decimal(3,1) unsigned,
    g7_sva decimal(3,1) unsigned,
    g8_sva decimal(3,1) unsigned,
    g9_sva decimal(3,1) unsigned,
    -- Engelska
    g6_en decimal(3,1) unsigned,
    g7_en decimal(3,1) unsigned,
    g8_en decimal(3,1) unsigned,
    g9_en decimal(3,1) unsigned,
    -- Matematik
    g6_ma decimal(3,1) unsigned,
    g7_ma decimal(3,1) unsigned,
    g8_ma decimal(3,1) unsigned,
    g9_ma decimal(3,1) unsigned,

    -- Biologi
    g6_bi decimal(3,1) unsigned,
    g7_bi decimal(3,1) unsigned,
    g8_bi decimal(3,1) unsigned,
    g9_bi decimal(3,1) unsigned,
    -- Fysik
    g6_fy decimal(3,1) unsigned,
    g7_fy decimal(3,1) unsigned,
    g8_fy decimal(3,1) unsigned,
    g9_fy decimal(3,1) unsigned,
    -- Kemi
    g6_ke decimal(3,1) unsigned,
    g7_ke decimal(3,1) unsigned,
    g8_ke decimal(3,1) unsigned,
    g9_ke decimal(3,1) unsigned,
    -- Teknik
    g6_tk decimal(3,1) unsigned,
    g7_tk decimal(3,1) unsigned,
    g8_tk decimal(3,1) unsigned,
    g9_tk decimal(3,1) unsigned,

    -- Geografi
    g6_ge decimal(3,1) unsigned,
    g7_ge decimal(3,1) unsigned,
    g8_ge decimal(3,1) unsigned,
    g9_ge decimal(3,1) unsigned,
    -- Historia
    g6_hi decimal(3,1) unsigned,
    g7_hi decimal(3,1) unsigned,
    g8_hi decimal(3,1) unsigned,
    g9_hi decimal(3,1) unsigned,
    -- Religion
    g6_re decimal(3,1) unsigned,
    g7_re decimal(3,1) unsigned,
    g8_re decimal(3,1) unsigned,
    g9_re decimal(3,1) unsigned,
    -- Samhällskunskap
    g6_sh decimal(3,1) unsigned,
    g7_sh decimal(3,1) unsigned,
    g8_sh decimal(3,1) unsigned,
    g9_sh decimal(3,1) unsigned,

    -- Bild
    g6_bd decimal(3,1) unsigned,
    g7_bd decimal(3,1) unsigned,
    g8_bd decimal(3,1) unsigned,
    g9_bd decimal(3,1) unsigned,
    -- Hem- och konsumentkunskap
    g6_hkk decimal(3,1) unsigned,
    g7_hkk decimal(3,1) unsigned,
    g8_hkk decimal(3,1) unsigned,
    g9_hkk decimal(3,1) unsigned,
    -- Idrott och hälsa
    g6_idh decimal(3,1) unsigned,
    g7_idh decimal(3,1) unsigned,
    g8_idh decimal(3,1) unsigned,
    g9_idh decimal(3,1) unsigned,
    -- Musik
    g6_mu decimal(3,1) unsigned,
    g7_mu decimal(3,1) unsigned,
    g8_mu decimal(3,1) unsigned,
    g9_mu decimal(3,1) unsigned,
    -- Slöjd
    g6_sl decimal(3,1) unsigned,
    g7_sl decimal(3,1) unsigned,
    g8_sl decimal(3,1) unsigned,
    g9_sl decimal(3,1) unsigned,

    -- Moderna språk
    g6_mspr decimal(3,1) unsigned,
    g7_mspr decimal(3,1) unsigned,
    g8_mspr decimal(3,1) unsigned,
    g9_mspr decimal(3,1) unsigned,

    primary key(ssn)
);

load data local infile '/home/fredrik/projects/python/pycharm-projects/tensorenv/csvnonames-utf-8.csv' into table sg fields terminated by ';' lines terminated by '\n' ignore 1 rows;
alter table sg add id int not null auto_increment primary key;

insert into clean(ssn, sex, mothertongue) select distinct ssn, sex, mothertongue from sg;
--update sg,clean set clean.sex=sg.sex where sg.ssn=clean.ssn;
--update sg,clean set clean.mothertongue=sg.mothertongue where sg.ssn=clean.ssn;

-- Sv, En, Ma

update clean,sg set clean.g6_sv=sg.points where sg.ssn=clean.ssn and sg.subject = "Svenska" and sg.year = "6";
update clean,sg set clean.g7_sv=sg.points where sg.ssn=clean.ssn and sg.subject = "Svenska" and sg.year = "7";
update clean,sg set clean.g8_sv=sg.points where sg.ssn=clean.ssn and sg.subject = "Svenska" and sg.year = "8";
update clean,sg set clean.g9_sv=sg.points where sg.ssn=clean.ssn and sg.subject = "Svenska" and sg.year = "9";

update clean,sg set clean.g6_sva=sg.points where sg.ssn=clean.ssn and sg.subject like "Svenska som%" and sg.year = "6";
update clean,sg set clean.g7_sva=sg.points where sg.ssn=clean.ssn and sg.subject like "Svenska som%" and sg.year = "7";
update clean,sg set clean.g8_sva=sg.points where sg.ssn=clean.ssn and sg.subject like "Svenska som%" and sg.year = "8";
update clean,sg set clean.g9_sva=sg.points where sg.ssn=clean.ssn and sg.subject like "Svenska som%" and sg.year = "9";

update clean,sg set clean.g6_en=sg.points where sg.ssn=clean.ssn and sg.subject = "Engelska" and sg.year = "6";
update clean,sg set clean.g7_en=sg.points where sg.ssn=clean.ssn and sg.subject = "Engelska" and sg.year = "7";
update clean,sg set clean.g8_en=sg.points where sg.ssn=clean.ssn and sg.subject = "Engelska" and sg.year = "8";
update clean,sg set clean.g9_en=sg.points where sg.ssn=clean.ssn and sg.subject = "Engelska" and sg.year = "9";

update clean,sg set clean.g6_ma=sg.points where sg.ssn=clean.ssn and sg.subject = "Matematik" and sg.year = "6";
update clean,sg set clean.g7_ma=sg.points where sg.ssn=clean.ssn and sg.subject = "Matematik" and sg.year = "7";
update clean,sg set clean.g8_ma=sg.points where sg.ssn=clean.ssn and sg.subject = "Matematik" and sg.year = "8";
update clean,sg set clean.g9_ma=sg.points where sg.ssn=clean.ssn and sg.subject = "Matematik" and sg.year = "9";

-- NO

update clean,sg set clean.g6_bi=sg.points where sg.ssn=clean.ssn and sg.subject = "Biologi" and sg.year = "6";
update clean,sg set clean.g7_bi=sg.points where sg.ssn=clean.ssn and sg.subject = "Biologi" and sg.year = "7";
update clean,sg set clean.g8_bi=sg.points where sg.ssn=clean.ssn and sg.subject = "Biologi" and sg.year = "8";
update clean,sg set clean.g9_bi=sg.points where sg.ssn=clean.ssn and sg.subject = "Biologi" and sg.year = "9";

update clean,sg set clean.g6_fy=sg.points where sg.ssn=clean.ssn and sg.subject = "Fysik" and sg.year = "6";
update clean,sg set clean.g7_fy=sg.points where sg.ssn=clean.ssn and sg.subject = "Fysik" and sg.year = "7";
update clean,sg set clean.g8_fy=sg.points where sg.ssn=clean.ssn and sg.subject = "Fysik" and sg.year = "8";
update clean,sg set clean.g9_fy=sg.points where sg.ssn=clean.ssn and sg.subject = "Fysik" and sg.year = "9";

update clean,sg set clean.g6_ke=sg.points where sg.ssn=clean.ssn and sg.subject = "Kemi" and sg.year = "6";
update clean,sg set clean.g7_ke=sg.points where sg.ssn=clean.ssn and sg.subject = "Kemi" and sg.year = "7";
update clean,sg set clean.g8_ke=sg.points where sg.ssn=clean.ssn and sg.subject = "Kemi" and sg.year = "8";
update clean,sg set clean.g9_ke=sg.points where sg.ssn=clean.ssn and sg.subject = "Kemi" and sg.year = "9";

update clean,sg set clean.g6_tk=sg.points where sg.ssn=clean.ssn and sg.subject = "Teknik" and sg.year = "6";
update clean,sg set clean.g7_tk=sg.points where sg.ssn=clean.ssn and sg.subject = "Teknik" and sg.year = "7";
update clean,sg set clean.g8_tk=sg.points where sg.ssn=clean.ssn and sg.subject = "Teknik" and sg.year = "8";
update clean,sg set clean.g9_tk=sg.points where sg.ssn=clean.ssn and sg.subject = "Teknik" and sg.year = "9";

-- SO

update clean,sg set clean.g6_ge=sg.points where sg.ssn=clean.ssn and sg.subject = "Geografi" and sg.year = "6";
update clean,sg set clean.g7_ge=sg.points where sg.ssn=clean.ssn and sg.subject = "Geografi" and sg.year = "7";
update clean,sg set clean.g8_ge=sg.points where sg.ssn=clean.ssn and sg.subject = "Geografi" and sg.year = "8";
update clean,sg set clean.g9_ge=sg.points where sg.ssn=clean.ssn and sg.subject = "Geografi" and sg.year = "9";

update clean,sg set clean.g6_hi=sg.points where sg.ssn=clean.ssn and sg.subject = "Historia" and sg.year = "6";
update clean,sg set clean.g7_hi=sg.points where sg.ssn=clean.ssn and sg.subject = "Historia" and sg.year = "7";
update clean,sg set clean.g8_hi=sg.points where sg.ssn=clean.ssn and sg.subject = "Historia" and sg.year = "8";
update clean,sg set clean.g9_hi=sg.points where sg.ssn=clean.ssn and sg.subject = "Historia" and sg.year = "9";

update clean,sg set clean.g6_re=sg.points where sg.ssn=clean.ssn and sg.subject = "Religionskunskap" and sg.year = "6";
update clean,sg set clean.g7_re=sg.points where sg.ssn=clean.ssn and sg.subject = "Religionskunskap" and sg.year = "7";
update clean,sg set clean.g8_re=sg.points where sg.ssn=clean.ssn and sg.subject = "Religionskunskap" and sg.year = "8";
update clean,sg set clean.g9_re=sg.points where sg.ssn=clean.ssn and sg.subject = "Religionskunskap" and sg.year = "9";

update clean,sg set clean.g6_sh=sg.points where sg.ssn=clean.ssn and sg.subject like "Samh%" and sg.year = "6";
update clean,sg set clean.g7_sh=sg.points where sg.ssn=clean.ssn and sg.subject like "Samh%" and sg.year = "7";
update clean,sg set clean.g8_sh=sg.points where sg.ssn=clean.ssn and sg.subject like "Samh%" and sg.year = "8";
update clean,sg set clean.g9_sh=sg.points where sg.ssn=clean.ssn and sg.subject like "Samh%" and sg.year = "9";

-- Praktiska ämnen

update clean,sg set clean.g6_bd=sg.points where sg.ssn=clean.ssn and sg.subject = "Bild" and sg.year = "6";
update clean,sg set clean.g7_bd=sg.points where sg.ssn=clean.ssn and sg.subject = "Bild" and sg.year = "7";
update clean,sg set clean.g8_bd=sg.points where sg.ssn=clean.ssn and sg.subject = "Bild" and sg.year = "8";
update clean,sg set clean.g9_bd=sg.points where sg.ssn=clean.ssn and sg.subject = "Bild" and sg.year = "9";

update clean,sg set clean.g6_hkk=sg.points where sg.ssn=clean.ssn and sg.subject = "Hem- och konsumentkunskap" and sg.year = "6";
update clean,sg set clean.g7_hkk=sg.points where sg.ssn=clean.ssn and sg.subject = "Hem- och konsumentkunskap" and sg.year = "7";
update clean,sg set clean.g8_hkk=sg.points where sg.ssn=clean.ssn and sg.subject = "Hem- och konsumentkunskap" and sg.year = "8";
update clean,sg set clean.g9_hkk=sg.points where sg.ssn=clean.ssn and sg.subject = "Hem- och konsumentkunskap" and sg.year = "9";

update clean,sg set clean.g6_idh=sg.points where sg.ssn=clean.ssn and sg.subject like "Idrott%" and sg.year = "6";
update clean,sg set clean.g7_idh=sg.points where sg.ssn=clean.ssn and sg.subject like "Idrott%" and sg.year = "7";
update clean,sg set clean.g8_idh=sg.points where sg.ssn=clean.ssn and sg.subject like "Idrott%" and sg.year = "8";
update clean,sg set clean.g9_idh=sg.points where sg.ssn=clean.ssn and sg.subject like "Idrott%" and sg.year = "9";

update clean,sg set clean.g6_mu=sg.points where sg.ssn=clean.ssn and sg.subject = "Musik" and sg.year = "6";
update clean,sg set clean.g7_mu=sg.points where sg.ssn=clean.ssn and sg.subject = "Musik" and sg.year = "7";
update clean,sg set clean.g8_mu=sg.points where sg.ssn=clean.ssn and sg.subject = "Musik" and sg.year = "8";
update clean,sg set clean.g9_mu=sg.points where sg.ssn=clean.ssn and sg.subject = "Musik" and sg.year = "9";

update clean,sg set clean.g6_sl=sg.points where sg.ssn=clean.ssn and sg.subject like "Sl%" and sg.year = "6";
update clean,sg set clean.g7_sl=sg.points where sg.ssn=clean.ssn and sg.subject like "Sl%" and sg.year = "7";
update clean,sg set clean.g8_sl=sg.points where sg.ssn=clean.ssn and sg.subject like "Sl%" and sg.year = "8";
update clean,sg set clean.g9_sl=sg.points where sg.ssn=clean.ssn and sg.subject like "Sl%" and sg.year = "9";

-- Moderna språk

update clean,sg set clean.g6_mspr=sg.points where sg.ssn=clean.ssn and (sg.subject = "Tyska" or sg.subject = "Franska" or sg.subject = "Spanska") and sg.year = "6";
update clean,sg set clean.g7_mspr=sg.points where sg.ssn=clean.ssn and (sg.subject = "Tyska" or sg.subject = "Franska" or sg.subject = "Spanska") and sg.year = "7";
update clean,sg set clean.g8_mspr=sg.points where sg.ssn=clean.ssn and (sg.subject = "Tyska" or sg.subject = "Franska" or sg.subject = "Spanska") and sg.year = "8";
update clean,sg set clean.g9_mspr=sg.points where sg.ssn=clean.ssn and (sg.subject = "Tyska" or sg.subject = "Franska" or sg.subject = "Spanska") and sg.year = "9";

-- create view full_ma as select * from clean where g6_ma is not null and g7_ma is not null and g8_ma is not null and g9_ma is not null;
