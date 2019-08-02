create table user (id int not null primary key unique auto_increment, \
                   name varchar(128) not null, \
                   notes text);

create table comp (sn varchar(128) not null primary key unique, \
                   parent_sn varchar(128), \
                   comp_type_id int not null, \
                   comp_type_rev_id int, \
                   user_id int not null, \
                   date_added datetime not null, \
                   notes text);

create table comp_type (id int not null primary key unique auto_increment, \
                        name varchar(128) not null, \
                        description text);

create table comp_type_rev (id int not null primary key unique auto_increment, \
                            comp_type_id int not null, \
                            name varchar(128) not null, \
                            description text);
create table comp_history(id int not null primary key unique auto_increment, \
                          comp_sn varchar(128) not null, \
                          user_id int not null, \
                          date_added datetime not null, \
                          ref text not null, \
                          notes text);
create table comp_testdata(id int not null primary key unique auto_increment, \
                           comp_sn varchar(128) not null, \
                           user_id int not null, \
                           date_added datetime not null, \
                           repo_path varchar(1024) not null, \
                           notes text);
CREATE TABLE prop_type(id INT NOT NULL PRIMARY KEY AUTO_INCREMENT, \
                       comp_type_id INT NOT NULL, \
                       name varchar(128) NOT NULL, \
                       units VARCHAR(128), \
                       notes VARCHAR(128));
create table layout(id int not null primary key unique auto_increment, \
                    name varchar(128) not null, \
                    frozen enum("Y", "N") not null, \
                    user_id int not null, \
                    notes text);
create table layout_conn(layout_id int not null, \
                         comp_sn1 varchar(128) not null, \
                         comp_sn2 varchar(128) not null, \
                         user_id int not null, \
                         notes text, \
                         index id (layout_id, comp_sn1, comp_sn2));
create table layout_prop(id int not null primary key unique auto_increment, \
                         layout_id int not null, \
                         comp_sn varchar(128) not null, \
                         user_id int not null, \
                         prop_type_id INT NOT NULL, \
                         value varchar(1024) not null, \
                         notes text);
create table layout_timestamp(id int not null primary key unique \
                                 auto_increment, \
                              layout_id int not null,
                              change_start datetime not null, \
                              change_end datetime not null, \
                              user_id int not null, \
                              notes text);

create table archive_acq(acq_id int not null primary key unique \
                         auto_increment, \
                         name varchar(64) not null, \
                         instrument varchar(64) not null, \
                         start_time double not null, \
                         finish_time double not null, \
                         notes text);

create table archive_file(acq_id int not null, \
                          file_id int not null, \
                          name varchar(64) not null, \
                          file_start double not null, \
                          file_end double not null, \
                          frames int not null, \
                          size_b int not null, \
                          format varchar(64) not null, \
                          md5sum varchar(64), \
                          index id (acq_id, file_id));
