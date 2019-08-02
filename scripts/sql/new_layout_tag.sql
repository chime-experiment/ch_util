RENAME TABLE layout_tag TO global_flag;
CREATE TABLE global_flag_category (id INT NOT NULL PRIMARY KEY AUTO_INCREMENT, \
                                  name VARCHAR(255) NOT NULL, \
                                  notes VARCHAR(65000));
ALTER TABLE global_flag ADD COLUMN category_id INT AFTER id, \
                       ADD FOREIGN KEY (category_id) \
                           REFERENCES global_flag_category(id);
INSERT INTO global_flag_category VALUES (0, "old style layout", "The original " \
    "layout database, before the current event-driven model, had monolithic " \
    "layouts with unique names; these layout tags preserve that information.");
INSERT INTO global_flag_category VALUES (0, "pass", "For defining discrete " \
    "passes of the data for analysis.");
UPDATE global_flag SET category_id = 1 WHERE id < 9285;
UPDATE global_flag SET category_id = 2 WHERE id >= 9285;
ALTER TABLE global_flag MODIFY COLUMN category_id INT NOT NULL;
ALTER TABLE global_flag ADD COLUMN severity \
            ENUM("comment", "warning", "severe") NOT NULL AFTER category_id;
ALTER TABLE global_flag ADD COLUMN inst_id INT AFTER severity, \
                       ADD FOREIGN KEY (inst_id) \
                           REFERENCES archiveinst(id);
UPDATE global_flag SET name = CONCAT(layout_id, " \&ndash; ", name) 
       WHERE id < 9285;
ALTER TABLE global_flag DROP COLUMN layout_id;
UPDATE user_permission_type SET name = "global_flag", \
       long_name = "set global flags" WHERE id = 6;
UPDATE event_type SET name = "global_flag", human_name = "global flag", \
       notes = "Denotes a general flag for the whole layout." where id = 7;
