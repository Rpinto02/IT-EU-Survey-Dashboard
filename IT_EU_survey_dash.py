import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


data_18 = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/IT_Salary_Survey_EU_18-20/Survey_2018.csv')
data_19 = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/IT_Salary_Survey_EU_18-20/Survey_2019.csv')
data_20 = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/IT_Salary_Survey_EU_18-20/Survey_2020.csv')

data_20_languague = data_20.copy()
data_20_languague ['Your main technology / programming language'] = data_20_languague ['Your main technology / programming language'].str.lower()
data_20_languague ['Your main technology / programming language'] = data_20_languague ['Your main technology / programming language'].str.replace('power bi','powerbi')
data_20_languague ['Your main technology / programming language'] = data_20_languague ['Your main technology / programming language'].str.replace('c +','c+')
data_20_languague ['Your main technology / programming language'] = data_20_languague ['Your main technology / programming language'].str.replace('pyrhon','python')
data_20_languague ['Your main technology / programming language'] = data_20_languague ['Your main technology / programming language'].str.replace('pythin','python')
data_20_languague ['Your main technology / programming language'] = data_20_languague ['Your main technology / programming language'].str.replace('kubrrnetes','kubernetes')
data_20_languague ['Your main technology / programming language'] = data_20_languague ['Your main technology / programming language'].str.replace('kuberenetes','kubernetes')
data_20_languague ['Your main technology / programming language'] = data_20_languague ['Your main technology / programming language'].str.replace('javscript','javascript')
data_20_languague ['Your main technology / programming language'] = data_20_languague ['Your main technology / programming language'].str.replace('.', ' ').str.replace(',', ' ').str.replace('/',' ')
data_20_languague ['Your main technology / programming language'] = data_20_languague ['Your main technology / programming language'].str.replace('&', '').str.replace('(', '').str.replace(')','')
data_20_languague ['Your main technology / programming language'] = data_20_languague ['Your main technology / programming language'].str.replace('missing', '0').str.replace('-', '0').str.replace('--','0')
data_20_languague ['Your main technology / programming language'] = data_20_languague ['Your main technology / programming language'].str.replace('none', '0').str.replace('nothing', '0')
data_20_languague ['Your main technology / programming language'] = data_20_languague ['Your main technology / programming language'].str.replace('*', '')
main_language_20 = data_20_languague['Your main technology / programming language'].str.split(expand=True,)

main_language_20_count = main_language_20.apply(pd.Series.value_counts)
main_language_20_count['total'] = main_language_20_count.sum(axis=1)
main_language_20_count = main_language_20_count.reset_index()
main_language_20_count.drop(columns=[0, 1, 2, 3, 4, 5, 6],inplace=True)
main_language_20_count.rename(columns={'index':'tech','total':'frequency'},inplace=True)
main_language_20_count.drop([0, 1, 2],inplace=True)
main_language_20_count.loc[main_language_20_count.frequency < 3, "tech"] = 'others'
main_language_20_count = main_language_20_count[main_language_20_count.tech != 'others']
#146 is the frequency of others
main_language_20_count = main_language_20_count.append({'tech': 'others','frequency':146},ignore_index=True)
main_language_20_count = main_language_20_count.sort_values(by='frequency')

layout = go.Layout(
    autosize=False,
    width=1000,
    height=2000)
fig_tech = go.Figure([go.Bar(x=main_language_20_count.frequency, 
                        y=main_language_20_count.tech,
                        text=main_language_20_count.frequency,
                        orientation='h',
                        width=0.9,
                        textfont=dict(family='Arial',
                                            size=16)
                       )],layout=layout)

fig_tech.update_layout(barmode='group', bargap=0.7,bargroupgap=0.0)
fig_tech.update_traces(texttemplate='%{text}', textposition='outside')

fig_tech.update_layout(
    xaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=False,

    ),
    yaxis=dict(
        showgrid=False,
        showline=False,
        zeroline=False
        ),
yaxis_tickfont_size=16)

text_tech = 'others represent all the languages/techs that showed a frequency of 2 or less'
    
fig_tech.update_layout(
    title_text="Frequency of technology used per person in 2020",
    annotations=[dict(text=text_tech, y=-1, font_size=15, showarrow=False, xref="paper")],
    showlegend=False,
    plot_bgcolor='white')




cleaned_salary_data_18 = data_18.loc[~data_18.loc[:,'Current Salary'].isna()]
salary_19 = data_19.loc[~data_19.loc[:,'Yearly brutto salary (without bonus and stocks)'].isna()]
salary_20 = data_20.loc[data_20.loc[:,'Total years of experience']!='MISSING']
salary_20.loc[:,'Total years of experience'] = salary_20.loc[:,'Total years of experience'].copy().apply(lambda x: x.replace(',','.') if isinstance(x, str) else x)
salary_20.loc[:,'Total years of experience'] = salary_20.loc[:,'Total years of experience'].copy().str.replace('less than year','0')
salary_20.loc[:,'Total years of experience'] = salary_20.loc[:,'Total years of experience'].copy().str.replace(r'[^\d.]+', '')
salary_20.loc[:,'Total years of experience'] = salary_20.loc[:,'Total years of experience'].copy().astype(float)
salary_20 = salary_20.loc[salary_20.loc[:,'Total years of experience']<100].copy()

salary_18 = cleaned_salary_data_18.groupby(['Years of experience'])
salary_18 = salary_18.filter(lambda x: len(x) > 3)

salary_18_over3freq = salary_18


salary_19_over3freq = salary_19.groupby(['Years of experience'])
salary_19_over3freq = salary_19_over3freq.filter(lambda x: len(x) > 3)

salary_20_over3freq = salary_20.groupby(['Total years of experience'])
salary_20_over3freq = salary_20_over3freq.filter(lambda x: len(x) > 3)

x_18 = salary_18_over3freq.groupby(['Years of experience'])['Current Salary'].median().index.tolist()
y_18 = salary_18_over3freq.groupby(['Years of experience'])['Current Salary'].median().round(2).tolist()


x_19 = salary_19_over3freq.groupby(['Years of experience'])['Yearly brutto salary (without bonus and stocks)'].median().index.tolist()
y_19 = salary_19_over3freq.groupby(['Years of experience'])['Yearly brutto salary (without bonus and stocks)'].median().round(2).tolist()



x_20 =salary_20_over3freq.groupby(['Total years of experience'])['Yearly brutto salary (without bonus and stocks) in EUR'].median().index.tolist()
y_20 =salary_20_over3freq.groupby(['Total years of experience'])['Yearly brutto salary (without bonus and stocks) in EUR'].median().round(2).tolist()


fig_salary = go.Figure()
fig_salary.add_trace(go.Scatter(x=x_18, y=y_18,
                    mode='lines',
                    line=dict(color='rgb(164,108,183)'),
                    name='lines'))

fig_salary.add_trace(go.Scatter(x=x_19, y=y_19,
                    mode='lines',
                    line=dict(color='rgb(203,106,73)'),
                    name='lines'))

fig_salary.add_trace(go.Scatter(x=x_20, y=y_20,
                    mode='lines',
                    line=dict(color='rgb(122,164,87)'),
                    name='lines'))
# max min points
fig_salary.add_trace(go.Scatter(x=[x_18[y_18.index(max(y_18))]],
                         y=[max(y_18)],
                         mode='markers+text',
                         marker=dict(color='rgb(164,108,183)'),
                        text=str(int(round(max(y_18),0)/1000))+'k',
                         textposition='top right',
                         textfont=dict(family='Arial',
                                            size=13,
                                           color='rgb(164,108,183)')
                        ))

fig_salary.add_trace(go.Scatter(x=[x_18[y_18.index(min(y_18))]],
                         y=[min(y_18)],
                         mode='markers+text',
                         marker=dict(color='rgb(164,108,183)'),
                        text=str(int(round(min(y_18),0)/1000))+'k',
                         textposition='bottom right',
                         textfont=dict(family='Arial',
                                            size=13,
                                           color='rgb(164,108,183)')
                        ))

fig_salary.add_trace(go.Scatter(x=[x_19[y_19.index(max(y_19))]],
                         y=[max(y_19)],
                         mode='markers+text',
                         marker=dict(color='rgb(203,106,73)'),
                        text=str(int(round(max(y_19),0)/1000))+'k',
                         textposition='top right',
                         textfont=dict(family='Arial',
                                            size=13,
                                           color='rgb(203,106,73)')
                        ))

fig_salary.add_trace(go.Scatter(x=[x_19[y_19.index(min(y_19))]],
                         y=[min(y_19)],
                         mode='markers+text',
                         marker=dict(color='rgb(203,106,73)'),
                        text=str(int(round(min(y_19),0)/1000))+'k',
                         textposition='top left',
                         textfont=dict(family='Arial',
                                            size=13,
                                           color='rgb(203,106,73)')
                        ))

fig_salary.add_trace(go.Scatter(x=[x_20[y_20.index(max(y_20))]],
                         y=[max(y_20)],
                         mode='markers+text',
                         marker=dict(color='rgb(122,164,87)'),
                        text=str(int(round(max(y_20),0)/1000))+'k',
                         textposition='top right',
                         textfont=dict(family='Arial',
                                            size=13,
                                           color='rgb(122,164,87)')
                        ))

fig_salary.add_trace(go.Scatter(x=[x_20[y_20.index(min(y_20))]],
                         y=[min(y_20)],
                         mode='markers+text',
                         marker=dict(color='rgb(122,164,87)'),
                        text=str(int(round(min(y_20),0)/1000))+'k',
                         textposition='bottom right',
                         textfont=dict(family='Arial',
                                            size=13,
                                           color='rgb(122,164,87)')
                        ))

y_18= np.array(y_18)
y_19= np.array(y_19)
y_20= np.array(y_20)
last_x18 = x_18[int(np.argwhere(y_18 == y_18[-1])[-1])]
last_x19 = x_19[int(np.argwhere(y_19 == y_19[-1])[-1])]
last_x20 = x_20[int(np.argwhere(y_20 == y_20[-1])[-1])]

last_x=[last_x18,last_x19,last_x20]
last_y=[y_18[-1],y_19[-1],y_20[-1]]

labels = ['2018', '2019', '2020']
colors = ['rgb(164,108,183)', 'rgb(203,106,73)', 'rgb(122,164,87)']


annotations = []


# Adding labels
for x, y, label, color in zip(last_x, last_y, labels, colors):
    annotations.append(dict(x=x+0.1, y=y,
                                  text=label,
                             xanchor='left', yanchor='middle',
                                  font=dict(family='Arial',
                                            size=15,
                                           color=color),
                            
                                  showarrow=False))
    
fig_salary.update_layout(annotations=annotations)

fig_salary.update_layout(
    title_text="Total years of experience vs the current salary",
    xaxis_title='Years of experience',
    yaxis_title='Current Salary per year in EUR',
    showlegend=False,
    plot_bgcolor='white')







labels_gender = data_20['Gender'].unique().tolist()
del labels_gender[2]
values_gender = data_20['Gender'].value_counts().values.tolist()

text = 'Men represent  <br> 138% more than <br> women.'

palette = ['rgb(104,122,210)','rgb(186,73,91)','rgb(80,180,123)']

fig_gender = go.Figure(data=[go.Pie(labels=labels_gender, values=values_gender, textinfo='label+percent', 
                             marker_colors=palette)])
fig_gender.update_layout(
    title_text="Gender ratio of the respondents in the year 2020",
    annotations=[dict(text=text, x=1.25, y=0.1, font_size=20, showarrow=False)])

gender_gap_18 = data_18.loc[~data_18.Gender.isna()]
gender_gap_19 = data_19
gender_gap_20 = data_20.loc[(data_20.Gender!='MISSING') & (data_20.Gender!='Diverse')]

gender_gap_18_cleaned = gender_gap_18.loc[~gender_gap_18['Current Salary'].isna()]
gender_gap_19_cleaned = gender_gap_19.loc[~gender_gap_19['Yearly brutto salary (without bonus and stocks)'].isna()]
gender_gap_20_cleaned = gender_gap_20

male_18 = gender_gap_18_cleaned.loc[gender_gap_18_cleaned.Gender=='M']
female_18 = gender_gap_18_cleaned.loc[gender_gap_18_cleaned.Gender=='F']
male_19 = gender_gap_19_cleaned.loc[gender_gap_19_cleaned.Gender=='Male']
female_19 = gender_gap_19_cleaned.loc[gender_gap_19_cleaned.Gender=='Female']
male_20 = gender_gap_20_cleaned.loc[gender_gap_20_cleaned.Gender=='Male']
female_20 = gender_gap_20_cleaned.loc[gender_gap_20_cleaned.Gender=='Female']

female_20 = female_20.loc[female_20['Yearly brutto salary (without bonus and stocks) in EUR']<150000]
male_20 = male_20.loc[male_20['Yearly brutto salary (without bonus and stocks) in EUR']<200000]

salary_male = [male_18['Current Salary'].median(), male_19['Yearly brutto salary (without bonus and stocks)'].median(), male_20['Yearly brutto salary (without bonus and stocks) in EUR'].median()]
salary_female = [female_18['Current Salary'].median(), female_19['Yearly brutto salary (without bonus and stocks)'].median(), female_20['Yearly brutto salary (without bonus and stocks) in EUR'].median()]

years = np.array([2018, 2019, 2020])
salary_male = np.array(salary_male)
salary_female = np.array(salary_female)

color_male ='rgb(104,122,210)'
color_female = 'rgb(186,73,91)'

years = np.array([2018, 2019, 2020])
salary_male = np.array(salary_male)
salary_female = np.array(salary_female)

color_male ='rgb(104,122,210)'
color_female = 'rgb(186,73,91)'

fig_gender_gap = go.Figure()

fig_gender_gap.add_trace(go.Scatter(x=years, y=salary_male,
                    mode='lines',
                    line=dict(color=color_male),
                    name='lines'))

fig_gender_gap.add_trace(go.Scatter(x=years, y=salary_female,
                    mode='lines',
                    line=dict(color=color_female),
                    name='lines'))


# endpoints
fig_gender_gap.add_trace(go.Scatter(
        x=[years[0], years[-1]],
        y=[salary_male[0], salary_male[-1]],
        mode='markers',
        marker=dict(color=color_male),
    ))

fig_gender_gap.add_trace(go.Scatter(
        x=[years[0], years[-1]],
        y=[salary_female[0], salary_female[-1]],
        mode='markers',
        marker=dict(color=color_female),
    ))


annotations =[]
# Adding labels
last_years=[years[-1],years[-1]]
last_salary=[salary_male[-1],salary_female[-1]]

labels_last = [str(int(round(salary_male[-1]/1000,0)))+'k €',str(int(round(salary_female[-1]/1000,0)))+'k €']
colors = [color_male,color_female]

first_years=[years[0],years[0]]
first_salary=[salary_male[0],salary_female[0]]

labels_first = [str(int(round(salary_male[0]/1000,0)))+'k €',str(int(round(salary_female[0]/1000,0)))+'k €']
colors = [color_male,color_female]

labels_gender = ['Men', 'Women']

annotations = []


# Adding labels
for x, y, label, color in zip(last_years, last_salary, labels_last, colors):
    annotations.append(dict(x=x+0.05, y=y,
                                  text=label,
                             xanchor='left', yanchor='middle',
                                  font=dict(family='Arial',
                                            size=15,
                                           color=color),
                            
                                  showarrow=False))
    
for x, y, label, color in zip(first_years, first_salary, labels_first, colors):
    annotations.append(dict(x=x-0.3, y=y,
                                  text=label,
                             xanchor='left', yanchor='middle',
                                  font=dict(family='Arial',
                                            size=15,
                                           color=color),
                            
                                  showarrow=False))

for x, y, label, color in zip(first_years, first_salary, labels_gender, colors):
    annotations.append(dict(x=x, y=y+3500,
                                  text=label,
                             xanchor='left', yanchor='middle',
                                  font=dict(family='Arial',
                                            size=15,
                                           color=color),
                            
                                  showarrow=False))    


fig_gender_gap.update_xaxes(
    ticktext=["2018", "2019", "2020"],
    tickvals=[2018, 2019, 2020, years],
)

fig_gender_gap.update_yaxes(
    ticktext=["0 k €"],
    tickvals=[0]
)

fig_gender_gap.update_yaxes(tickprefix="€")
    
fig_gender_gap.update_yaxes(rangemode="tozero")
fig_gender_gap.update_layout(annotations=annotations)

fig_gender_gap.update_layout(
    title_text="Median salary over the years",
    showlegend=False,
    plot_bgcolor='white')


gender_18 = data_18.loc[~data_18.Gender.isna()]
gender_19 = data_19
gender_20 = data_20.loc[data_20.Gender!='MISSING']


gender = gender_18.Gender.value_counts().rename_axis('gender').reset_index(name='counts')
gender['gender'] = gender['gender'].str.replace('M', 'Male')
gender['gender'] = gender['gender'].str.replace('F', 'Female')
gender['year'] = 2018

df2 = gender_19.Gender.value_counts().rename_axis('gender').reset_index(name='counts')
df2['year'] = 2019
gender = gender.append(df2).reset_index(drop=True)


df3 = gender_20.Gender.value_counts().rename_axis('gender').reset_index(name='counts')
df3['year'] = 2020
gender = gender.append(df3).reset_index(drop=True)

gender_years = gender['year'].unique()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets) 




app.layout = html.Div(children=[
    # All elements from the top of the page
    html.Div([
        html.H1(children='IT EU Survey'),
        html.Div([
        html.Div([dcc.Dropdown(
        id='dropdown',
        options=[
           
            {'label': '2018', 'value': 2018},
            {'label': '2019', 'value': 2019},
            {'label': '2020', 'value': 2020},
            {'label': 'Total', 'value': 'Total'}
        ],
        value='Total',
        multi=False,
        clearable=False,
        style={'width':'35%'}
    ),
    html.Div([dcc.Graph(id='gender')])
        ], className="six columns"),

        html.Div([
            dcc.Graph(id='g2', figure=fig_gender_gap)
        ], className="six columns"),
    ], className="row")]),

    html.Div([
    
        dcc.Graph(
            id='g3',
            figure=fig_salary
        ),  
    ]),
    html.Div([
    
        dcc.Graph(
            id='g4',
            figure=fig_tech
        ),  
    ]),

])




@app.callback(
    dash.dependencies.Output(component_id='gender', component_property='figure'),
    [dash.dependencies.Input(component_id='dropdown', component_property='value')]
    )
def update_graph(dropdown):
    if dropdown == 'Total':
        gender_copy = gender
    else:
        gender_copy = gender.loc[gender['year'] == dropdown]

    palette = ['rgb(104,122,210)','rgb(186,73,91)','rgb(80,180,123)']

    fig_gender = px.pie(gender_copy, values='counts', names='gender',
             title='Gender ratio', color_discrete_sequence = palette)
    fig_gender.update_traces(textposition='inside', textinfo='percent+label')
   
    return fig_gender

#uncomment the next line to run localy    
#app.run_server(debug=True, use_reloader=False) 