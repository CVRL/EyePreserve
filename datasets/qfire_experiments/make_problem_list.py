import os

directory = './Problems/'

for visit in os.listdir(directory):
    visit_dir = os.path.join(directory, visit)
    for eyetype in os.listdir(visit_dir):
        visit_eyetype_dir = os.path.join(visit_dir, eyetype)
        with open(visit + '_' + eyetype + '.txt', 'w') as txtfile:
            for imagename in os.listdir(visit_eyetype_dir):
                txtfile.write(imagename+'\n')
            
