
window.extend = (obj) ->
  obj ||= {}
  for arg, idx in arguments 
    if idx > 0      
      for own name,s of arg
        if !obj[name]? || obj[name] != s
          obj[name] = s
  obj

window.defaults = (o) ->
  obj = {}

  for arg, idx in arguments by -1      
    for own name,s of arg
      obj[name] = s
  extend o, obj



window.alphabetical_compare = (a,b) ->
  if a.toLowerCase().trim() < b.toLowerCase().trim() 
    return -1 
  else 
    return 1




window.random = (min, max) -> Math.random() * (max - min) + min
window.randomColor = -> "rgba(#{random(0, 255)}, #{random(0, 255)}, #{random(0, 255)}, 0.5)"

window.generateColorPalette = (numColors) ->
  baseHue = random(0, 360);
  colors = [];
  hueStep = 360 / numColors;

  i = 0
  while i < numColors
    h = (baseHue + (i * hueStep)) % 360
    color = "hsl(#{h}, 60%, 50%)"
    colors.push(color)
    i += 1
  return colors





dom.JSONEditorWithSchema = ->
  state = retrieve('json_editor')
  song_config = retrieve("/song_config/#{state.song}")

  DIV 
    ref: 'editor_holder'

dom.JSONEditorWithSchema.up = dom.JSONEditorWithSchema.refresh = ->
  element = @refs.editor_holder.getDOMNode()

  state = retrieve('json_editor')

  song_config = retrieve("/song_config/#{state.song}")

  if !song_config.config
    return


  if !@initialized

    editor = new JSONEditor element,
      theme: 'spectre'
      iconlib: 'spectre'
      schema:
        $ref: "/library/library_schema.json"
        format: "categories"
      startval: song_config.config
      ajax: true

    @initialized = true

# JSONEditorTextArea          
#   id: 'json'
#   key: md5(subdomain.customizations) # update text area if subdomain.customizations changes elsewhere
#   json: subdomain.customizations
#   onChange: (val) => 
#     @local.stringified_current_value = val



dom.JSONEditorTextArea = ->
  DIV
    style: 
      height: '90vh'
      position: 'relative'

    if @local.has_changes
      BUTTON
        style:
          postion: 'fixed'
          left: 0
          top: 0
        onClick: =>
          @props.onSave?()
          @local.has_changes = false
          save @local
        'Save'

    DIV 
      ref: 'json_editor'
      style: 
        height: '90vh'





dom.JSONEditorTextArea.up = dom.JSONEditorTextArea.refresh = ->

  if !@initialized
    editor = new JSONEditor @refs.json_editor.getDOMNode(), 
      mode: 'code'
      modes: ['tree', 'code']
      sortObjectKeys: true
      mainMenuBar: false

      onChange: =>
        try 
          @props.onChange? editor.getText()
          @props.onChangeJSON?()
          @local.has_changes = true
          save @local


        catch e 
          console.log 'Got error', e



    editor.set @props.json

    @initialized = true

