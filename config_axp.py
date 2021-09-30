for ii in range(100):
    start_pos = tuple(tuple(all_free[randint(0,all_free.__len__()-1)]) for f in range(loop_number_of_agent[0]))
    print(start_pos)

data_file = open(f'{loop_number_of_agent}_agent_{datetime.now()}.csv', 'w', newline='\n')

writer = csv.writer(data_file, delimiter=',')
writer.writerow(
    ['map_name', 'start_state', 'time', 'h type', 'h_start', 'h_genarate', 'h_expend', 'number of max pivot',
     'use black list', 'genarate', 'expend', 'open is beter', 'new is beter', 'obs remove', 'cost'])

start_config_as_string = np.loadtxt(f'./config/{map_type}_{15}_agent_domain.csv', dtype=tuple,delimiter='\n')
all_start_config_as_tupel = [ast.literal_eval(i) for i in start_config_as_string]

all_start_config_as_tupel = [tuple((i[0],i[1],i[2],i[3],i[4],i[5],i[6],i[7],i[8],i[9],i[10],i[11],i[12],i[13],i[14],tuple(all_free[randint(0,all_free.__len__()-1)]))) for i in all_start_config_as_tupel]
for i in all_start_config_as_tupel:
    print(i)