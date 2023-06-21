# Jobbot

[jobbot.pro](jobbot.pro)

```yml
- name: Minnesota Works Candidates
  url: https://www.minnesotaworks.net/
  credentials:
    username: ${MINNESOTA_WORKS_CANDIDATES_USERNAME}
    password: ${MINNESOTA_WORKS_CANDIDATES_PASSWORD}
  tags:
    - category:candidates
    - location:minnesota

- name: Career Force
  url: https://www.careerforcemn.com/job-search
  # ?title=&location%5Btext%5D=&op=Search+jobs
  tags:
    - category:jobs
    - location:minnesota

- name: CAREERWise Minnesota State
  url: https://careerwise.minnstate.edu/
  tags:
    - category:jobs
    - location:minnesota

- name: Career One Stop
  url: https://www.careeronestop.org/JobSearch/job-search.aspx
  tags:
    - category:jobs
    - location:minnesota

- name: State of Minnesota
  url: https://mn.gov/mmb/careers/search-for-jobs/
  tags:
    - category:jobs
    - location:minnesota

- name: USA Jobs
  url: https://www.usajobs.gov/
  tags:
    - category:jobs

- name: University of Minnesota
  url: https://hr.umn.edu/
  tags:
    - category:jobs
    - location:minnesota

- name: Higher Education Recruitment Consortium
  url: https://www.hercjobs.org/regions/higher-ed-careers-minnesota/
  tags:
    - category:jobs
    - location:minnesota
```
