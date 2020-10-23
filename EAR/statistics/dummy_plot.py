import numpy as np
import matplotlib.pyplot as plt

# Yelp
ear_results = [0, 0.043, 0.26, 0.511, 0.62, 0.678, 0.762, 0.809,
                0.855, 0.895, 0.923, 0.942, 0.955, 0.963, 0.968]
crm_resutls = [0, 0.046, 0.187, 0.368, 0.485, 0.589, 0.667, 0.718,
                0.761, 0.799, 0.828, 0.848, 0.865, 0.877, 0.888]
sac_results = [0.002, 0.085, 0.122, 0.146, 0.172, 0.199, 0.237, 
                0.262, 0.276, 0.294, 0.308, 0.317, 0.327, 0.340,
                0.348]

ear_results = np.array(ear_results)
crm_resutls = np.array(crm_resutls)
sac_results = np.array(sac_results)

plt.figure()
plt.plot(range(15), ear_results, '-o', label='EAR')
plt.plot(range(15), crm_resutls, '-o', label='CRM')
plt.plot(range(15), sac_results, '-o', label='SAC')
plt.ylabel('Success Rate')
plt.xlabel('turns')
plt.legend()
plt.title('Yelp')
plt.savefig('yelp.pdf')

# lastFM
ear_results = [0, 0, 0.001, 0.003, 0.019, 0.047, 0.090, 0.139, 
                0.194, 0.246, 0.290, 0.334, 0.367, 0.399, 0.429]
crm_resutls = [0, 0.001, 0.002, 0.004, 0.011, 0.026, 0.052, 0.074,
                0.105, 0.128, 0.156, 0.185, 0.217, 0.244, 0.270]
sac_results = [0, 0, 0.001, 0.003, 0.004, 0.006, 0.008, 0.011, 0.015,
                0.02, 0.026, 0.032, 0.038, 0.046, 0.054]

ear_results = np.array(ear_results)
crm_resutls = np.array(crm_resutls)
sac_results = np.array(sac_results)

plt.figure()
plt.plot(range(15), ear_results, '-o', label='EAR')
plt.plot(range(15), crm_resutls, '-o', label='CRM')
plt.plot(range(15), sac_results, '-o', label='SAC')
plt.ylabel('Success Rate')
plt.xlabel('turns')
plt.legend()
plt.title('lastFM')
plt.savefig('lastfm.pdf')
