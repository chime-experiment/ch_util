-- RENAME TABLE comp_type TO component_type;
-- RENAME TABLE comp_type_rev TO component_type_rev;
DROP TABLE IF EXISTS layout_tag;
DROP TABLE IF EXISTS layout_note;
DROP TABLE IF EXISTS property;
DROP TABLE IF EXISTS property_component;
DROP TABLE IF EXISTS property_type;
DROP TABLE IF EXISTS connexion;
DROP TABLE IF EXISTS component_doc;
DROP TABLE IF EXISTS component_history;
DROP TABLE IF EXISTS component;
DROP TABLE IF EXISTS event;
DROP TABLE IF EXISTS event_type;
DROP TABLE IF EXISTS graph_obj;
DROP TABLE IF EXISTS predef_subgraph_spec_param;
DROP TABLE IF EXISTS predef_subgraph_spec;
DROP TABLE IF EXISTS user_permission;
DROP TABLE IF EXISTS user_permission_type;
DROP TABLE IF EXISTS timestamp;
DROP TABLE IF EXISTS external_repo;
DROP TABLE IF EXISTS component_history;
DROP TABLE IF EXISTS component_reference;

-- ALTER TABLE component_type MODIFY name VARCHAR(255);
-- ALTER TABLE component_type MODIFY notes VARCHAR(65000);
-- ALTER TABLE component_type_rev CHANGE comp_type_id type_id INT;
-- ALTER TABLE component_type_rev ADD FOREIGN KEY (type_id) \
--             REFERENCES component_type(id);
-- ALTER TABLE component_type_rev MODIFY name VARCHAR(255);
-- ALTER TABLE component_type_rev MODIFY notes VARCHAR(65000);

UPDATE layout_conn SET comp_sn1 = "ATMEGA49704949575721220150" \
       WHERE comp_sn1 = "49 70 49 49 57 57 21 22 0 15 0";

DELETE FROM layout_conn WHERE (comp_sn1 LIKE "ANT%" AND comp_sn2 LIKE "PL%") \
                           OR (comp_sn1 LIKE "PL%" AND comp_sn2 LIKE "ANT%") \
                           AND layout_id >= 1;

CREATE TABLE user_permission_type (id INT NOT NULL PRIMARY KEY UNIQUE \
                                      AUTO_INCREMENT, \
                                   name VARCHAR(255) NOT NULL, \
                                   long_name VARCHAR(65000));

INSERT INTO user_permission_type VALUES \
       (0, "predef_subgraph", "add/edit predefined subgraph templates");
INSERT INTO user_permission_type VALUES \
       (0, "connexion", "make/break connexions");
INSERT INTO user_permission_type VALUES \
       (0, "comp_avail", "add/remove components");
INSERT INTO user_permission_type VALUES \
       (0, "comp_info", "add/edit/remove component history/document events");
INSERT INTO user_permission_type VALUES \
       (0, "property", "add/remove/edit properties");
INSERT INTO user_permission_type VALUES \
       (0, "layout_tag", "tag layouts");
-- INSERT INTO user_permission_type VALUES \
--        (0, "layout_note", "make notes about the layout");
INSERT INTO user_permission_type VALUES \
       (0, "type_tables", "edit component type/rev & property type tables");
INSERT INTO user_permission_type VALUES \
       (0, "db_admin", "perform user administration");

CREATE TABLE user_permission (user_id INT UNSIGNED NOT NULL,
                              type_id INT NOT NULL,
                              FOREIGN KEY (user_id)
                                      REFERENCES chimewiki.user(user_id), \
                              FOREIGN KEY (type_id)
                                      REFERENCES user_permission_type(id), \
                              PRIMARY KEY id (user_id, type_id));

INSERT INTO user_permission SELECT user_id, id FROM chimewiki.user \
       JOIN user_permission_type \
       WHERE user_permission_type.name = "predef_subgraph";
INSERT INTO user_permission SELECT user_id, id FROM chimewiki.user \
       JOIN user_permission_type \
       WHERE user_permission_type.name = "connexion";
INSERT INTO user_permission SELECT user_id, id FROM chimewiki.user \
       JOIN user_permission_type \
       WHERE user_permission_type.name = "comp_avail";
INSERT INTO user_permission SELECT user_id, id FROM chimewiki.user \
       JOIN user_permission_type \
       WHERE user_permission_type.name = "comp_info";
INSERT INTO user_permission SELECT user_id, id FROM chimewiki.user \
       JOIN user_permission_type \
       WHERE user_permission_type.name = "property";
INSERT INTO user_permission SELECT user_id, id FROM chimewiki.user \
       JOIN user_permission_type \
       WHERE user_permission_type.name = "layout_tag";


CREATE TABLE event_type (id INT NOT NULL PRIMARY KEY UNIQUE AUTO_INCREMENT, \
                         name VARCHAR(255) NOT NULL, \
                         human_name VARCHAR(255) NOT NULL, \
                         assoc_table VARCHAR(255),
                         no_end ENUM("Y", "N") NOT NULL DEFAULT "N", \
                         require_name ENUM("Y", "N") NOT NULL DEFAULT "N", \
                         require_notes ENUM("Y", "N") NOT NULL DEFAULT "N", \
                         notes VARCHAR(255));

INSERT INTO event_type VALUES (0, "comp_avail", "component availability", \
                               "component", "N", \
                               "N", "N", "A component has been added to the " \
                               "experiment.");
INSERT INTO event_type VALUES (0, "connexion", "connexion", "connexion", \
       "N", "N", "N", \
       "Defines the period of time (possibly with no end) during which " \
       "a connexion exists.");
INSERT INTO event_type VALUES (0, "property", "component property",\
       "property", "N", "N", "N", \
       "Defines the period of time (possibly with no end) during which " \
       "a component property exists.");
INSERT INTO event_type VALUES (0, "perm_connexion", "permanent connexion", \
       "connexion", "Y", "N",\
       "N", "Defines a permanent connexion between components (what used to " \
       "be called parent-child components). For example, an antenna and its " \
       "polarisation output are permanently connected.");
INSERT INTO event_type VALUES (0, "comp_history", "component history", \
       "component_history", "N", "Y", \
       "N", "Tags a component with a comment or note.");
INSERT INTO event_type VALUES (0, "comp_doc", "component document", \
       "component", "Y", "N", \
       "N", "Tags a component with a reference to some document.");
INSERT INTO event_type VALUES (0, "layout_tag", "layout tag", NULL, "N", "N", \
       "N", "Creates a tag denoting a specific layout has started.");
-- INSERT INTO event_type VALUES (0, "layout_note", "layout note", NULL, "N", "Y",\
--       "N", "Creates a generic note about a layout.");

CREATE TABLE graph_obj (id INT NOT NULL PRIMARY KEY UNIQUE AUTO_INCREMENT);

CREATE TABLE timestamp (id INT NOT NULL PRIMARY KEY UNIQUE AUTO_INCREMENT, \
                        time DATETIME NOT NULL, \
                        entry_time DATETIME NOT NULL, \
                        user_id INT UNSIGNED NOT NULL, \
                        notes VARCHAR(65000), \
                        INDEX time_index (time), \
                        FOREIGN KEY (user_id) 
                                REFERENCES chimewiki.user(user_id));

CREATE TABLE event (id INT NOT NULL PRIMARY KEY UNIQUE AUTO_INCREMENT, \
                    active TINYINT NOT NULL DEFAULT 1, \
                    replaces_id INT, \
                    graph_obj_id INT NOT NULL, \
                    type_id INT NOT NULL, \
                    start_id INT NOT NULL, \
                    end_id INT, \
                    FOREIGN KEY (replaces_id) REFERENCES event(id), \
                    FOREIGN KEY (graph_obj_id) REFERENCES graph_obj(id), \
                    FOREIGN KEY (type_id) REFERENCES event_type(id), \
                    FOREIGN KEY (start_id) REFERENCES timestamp(id), \
                    FOREIGN KEY (end_id) REFERENCES timestamp(id), \
                    INDEX type_id_index (type_id),
                    INDEX time_index (start_id, end_id));

CREATE TABLE external_repo (id INT NOT NULL PRIMARY KEY UNIQUE AUTO_INCREMENT,
                            name VARCHAR(255) NOT NULL,
                            root VARCHAR(255) NOT NULL,
                            notes VARCHAR(65000));
INSERT INTO external_repo VALUES \
       (0, "hyper", "http://www.phas.ubc.ca/~chime/", NULL);

CREATE TABLE layout_tag (id INT NOT NULL PRIMARY KEY, \
                         layout_id INT NOT NULL UNIQUE AUTO_INCREMENT, \
                         name VARCHAR(255) NOT NULL, \
                         notes VARCHAR(65000), \
                         FOREIGN KEY (id) REFERENCES graph_obj(id));

-- CREATE TABLE layout_note (id INT NOT NULL PRIMARY KEY,
--                          subject VARCHAR(255) NOT NULL, \
--                          notes VARCHAR(65000), \
--                          FOREIGN KEY (id) REFERENCES graph_obj(id));

CREATE TABLE component (id INT NOT NULL PRIMARY KEY, \
                   sn VARCHAR(255) NOT NULL UNIQUE, \
                   type_id INT NOT NULL, \
                   type_rev_id INT,
                   FOREIGN KEY (id) REFERENCES graph_obj(id), \
                   INDEX sn_index (sn), \
                   FOREIGN KEY (type_id) REFERENCES component_type(id),
                   FOREIGN KEY (type_rev_id) REFERENCES component_type_rev(id));
DELETE FROM comp WHERE sn = "CAS001FLAT1";

CREATE TABLE component_history (id INT NOT NULL PRIMARY KEY, \
                                comp_sn VARCHAR(255) NOT NULL, \
                                notes VARCHAR(65000) NOT NULL, \
                                FOREIGN KEY (id) REFERENCES graph_obj(id), \
                                FOREIGN KEY (comp_sn) \
                                        REFERENCES component(sn));
                                

CREATE TABLE component_doc (id INT NOT NULL PRIMARY KEY UNIQUE 
                                AUTO_INCREMENT, \
                             comp_sn VARCHAR(255) NOT NULL, \
                             repo_id INT NOT NULL, \
                             ref VARCHAR(65000), \
                             FOREIGN KEY (id) REFERENCES graph_obj(id), \
                             FOREIGN KEY (comp_sn) \
                                     REFERENCES component(sn), \
                             FOREIGN KEY (repo_id) \
                                     REFERENCES external_repo(id));

CREATE TABLE connexion (id INT NOT NULL PRIMARY KEY, \
                        comp_sn1 VARCHAR(255) NOT NULL, \
                        comp_sn2 VARCHAR(255) NOT NULL, \
                        FOREIGN KEY (id) REFERENCES graph_obj(id), \
                        FOREIGN KEY (comp_sn1) REFERENCES component(sn), \
                        FOREIGN KEY (comp_sn2) REFERENCES component(sn), \
                        UNIQUE INDEX component_sn (comp_sn1, comp_sn2));


CREATE TABLE property_type (id INT NOT NULL PRIMARY KEY UNIQUE AUTO_INCREMENT, \
                        name VARCHAR(255) NOT NULL, \
                        units VARCHAR(255), \
                        regex VARCHAR(255), \
                        notes VARCHAR(255));

INSERT INTO property_type (SELECT 0, name, units, NULL, notes \
       FROM prop_type GROUP BY name);

UPDATE property_type SET regex = "^[+-]{0,1}((\\d+(\\.\\d*)?)|(\\.\\d+))$" \
       WHERE name IN ("attenuation", "dist_to_edge", "dist_to_n_end", \
                      "input_term", "latitude", "longitude", "roll", \
                      "termination");
UPDATE property_type SET regex = "^[NS]$" WHERE name = "slot_zero_pos";
UPDATE property_type SET regex = "^[NWES]{1,}$" \
       WHERE name IN ("hpol_orient", "pol1_orient", "pol2_orient", \
                      "vpol_orient");

CREATE TABLE property_component (prop_type_id INT NOT NULL, \
                                 comp_type_id INT NOT NULL, \
                                 FOREIGN KEY (prop_type_id) \
                                         REFERENCES property_type(id), \
                                 FOREIGN KEY (comp_type_id) \
                                         REFERENCES component_type(id), \
                                 PRIMARY KEY id (prop_type_id, comp_type_id));

INSERT INTO property_component (SELECT p2.id, comp_type_id FROM \
                                prop_type p1 \
                                JOIN property_type p2 ON p1.name = p2.name);

UPDATE layout_prop SET value = "0" WHERE prop_type_id IN (1, 10, 20) AND \
                                         value = "none";
UPDATE layout_prop SET value = "2.5" WHERE value = "~2.5";
                             
CREATE TABLE property (id INT NOT NULL PRIMARY KEY, \
                   comp_sn VARCHAR(128) NOT NULL, \
                   type_id INT NOT NULL, \
                   value VARCHAR(255) NOT NULL, \
                   FOREIGN KEY (id) REFERENCES graph_obj(id), \
                   FOREIGN KEY (comp_sn) REFERENCES component(sn), \
                   FOREIGN KEY (type_id) REFERENCES property_type(id),
                   INDEX component_sn_index(comp_sn, type_id));

CREATE TABLE predef_subgraph_spec(id INT NOT NULL UNIQUE PRIMARY KEY \
                       AUTO_INCREMENT, \
                       name VARCHAR(128) NOT NULL, \
                       start_type_id INT NOT NULL,
                       notes VARCHAR(65000),\
                       FOREIGN KEY (start_type_id) 
                               REFERENCES component_type(id));

CREATE TABLE predef_subgraph_spec_param(id INT NOT NULL UNIQUE PRIMARY KEY \
                             AUTO_INCREMENT, \
                             predef_subgraph_spec_id INT NOT NULL, \
                             type1_id INT NOT NULL, \
                             type2_id INT, \
                             action ENUM("T", "H", "O") NOT NULL, \
                             FOREIGN KEY (predef_subgraph_spec_id) \
                                     REFERENCES predef_subgraph_spec(id), \
                             FOREIGN KEY (type1_id) \
                                     REFERENCES component_type(id),
                             FOREIGN KEY (type2_id) \
                                     REFERENCES component_type(id),
                             UNIQUE INDEX (predef_subgraph_spec_id, type1_id, \
                                           type2_id, action));

INSERT INTO predef_subgraph_spec VALUES (0, "single antenna, no HK", 2, "A " \
                              "single " \
                              "antenna through to an ADC input; housekeeping " \
                              "part is ignored.");
INSERT INTO predef_subgraph_spec VALUES (0, "reflector to correlator, abbr.", 8,
                                         NULL);
INSERT INTO predef_subgraph_spec VALUES (0, "correlator to reflector, abbr.",
                                         38, NULL);
INSERT INTO predef_subgraph_spec VALUES (0, "HK mux to LNA",
                                         27, NULL);

INSERT INTO `predef_subgraph_spec_param` VALUES (0,1,8,NULL,'T'),(0,1,8,NULL,'H'),(0,1,9,NULL,'T'),(81,1,9,NULL,'H'),(0,1,11,NULL,'T'),(0,1,15,NULL,'T'),(0,1,15,NULL,'H'),(0,1,16,NULL,'H'),(0,1,25,NULL,'T'),(0,1,25,NULL,'H'),(0,1,26,NULL,'H'),(0,1,26,27,'O'),(0,1,27,NULL,'H'),(0,1,38,NULL,'T'),(0,1,43,38,'O'),(0,1,44,43,'O'),(0,2,1,NULL,'H'),(0,2,3,NULL,'H'),(0,2,4,NULL,'H'),(0,2,5,NULL,'H'),(0,2,6,NULL,'H'),(0,2,8,16,'O'),(0,2,9,NULL,'H'),(0,2,11,10,'O'),(0,2,12,NULL,'H'),(0,2,13,NULL,'H'),(0,2,14,NULL,'H'),(0,2,15,NULL,'T'),(0,2,15,NULL,'H'),(0,2,16,NULL,'H'),(0,2,18,NULL,'H'),(0,2,19,NULL,'H'),(0,2,20,NULL,'H'),(0,2,21,NULL,'H'),(0,2,25,NULL,'T'),(0,2,25,NULL,'H'),(0,2,27,NULL,'H'),(0,2,37,NULL,'H'),(0,2,38,NULL,'T'),(0,2,39,NULL,'H'),(0,2,40,NULL,'H'),(0,2,41,NULL,'H'),(0,2,42,NULL,'H'),(0,2,43,38,'O'),(0,2,44,43,'O'),(0,2,49,NULL,'T'),(0,3,3,NULL,'H'),(0,3,4,NULL,'H'),(0,3,5,NULL,'H'),(0,3,6,NULL,'H'),(0,3,8,NULL,'T'),(0,3,8,16,'O'),(0,3,9,NULL,'H'),(0,3,9,7,'O'),(0,3,12,NULL,'H'),(0,3,13,NULL,'H'),(0,3,13,2,'O'),(0,3,14,NULL,'H'),(0,3,15,NULL,'T'),(0,3,15,NULL,'H'),(0,3,16,NULL,'T'),(0,3,16,NULL,'H'),(0,3,18,NULL,'H'),(0,3,19,NULL,'H'),(0,3,20,NULL,'H'),(0,3,21,NULL,'H'),(0,3,25,NULL,'T'),(0,3,25,NULL,'H'),(0,3,37,NULL,'H'),(0,3,39,NULL,'H'),(0,3,40,NULL,'H'),(0,3,41,NULL,'H'),(0,3,42,NULL,'H'),(0,4,1,NULL,'T'),(0,4,3,NULL,'T'),(0,4,3,NULL,'H'),(0,4,8,NULL,'T'),(0,4,9,NULL,'T'),(0,4,9,NULL,'H'),(0,4,14,NULL,'T'),(0,4,14,NULL,'H'),(0,4,26,NULL,'T'),(0,4,26,NULL,'H'),(0,4,41,NULL,'T'),(0,4,41,NULL,'H'),(0,4,50,NULL,'T'),(0,4,50,NULL,'H');
