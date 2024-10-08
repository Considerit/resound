
dom.BODY = -> 
  
  loc = retrieve('location')

  if !loc.url or loc.url == ''
    loc.path = '/songs'
    loc.url = 'songs'
    save loc
    return DIV null

  DIV 
    style: 
      fontFamily: '-apple-system, BlinkMacSystemFont, sans-serif'

    TOOLTIP()

    NAVBAR()

    if loc.url == 'songs' or loc.url == ''
      SONGS()

    else if loc.url == 'channels'
      CHANNELS()

    else if loc.url.startsWith('songs/')
      parts = loc.url.split('/')
      SONG
        song: decodeURI(parts[1])

dom.NAVBAR = ->
  loc = retrieve('location')

  current_path = '/' + loc.path.split('/')[1] 

  paths = [ ['/songs', 'Songs'], ['/channels', 'Channels'] ]

  asty = 
    color: 'black'
    textDecoration: 'none'
    fontWeight: 700
    display: 'inline-block'
    padding: '8px 24px'

  DIV 
    style:
      width: '100%'
      backgroundColor: '#f59568'

    for path in paths

      A
        style: extend {}, asty, 
          color: if path[0] == current_path then 'white' else 'black'
          backgroundColor: if path[0] == current_path then '#db5516' else '#f59568'

        href: path[0]
        path[1]



dom.SONGS = -> 
  all = retrieve('/songs').songs
  return DIV null if !all


  pinned = (s for s in all when retrieve("/song_config/#{s}").config?.pinned)

  DIV null,
    UL 
      style:
        marginTop: 50
        listStyle: 'none'
        paddingLeft: 60
      for song in pinned
        LI 
          style:
            display: 'flex'

          BUTTON
            title: 'Sync with Resoundio.com'
            style:
              marginRight: 18
            onClick: do(song) => =>
              save {
                'key': "/sync_with_resoundio/#{song}",
                'song': song
              }, ->
                bus.forget("/sync_with_resoundio/#{song}")
            
            I
              className: "glyphicon glyphicon-refresh"

          A
            href: "/songs/#{song}"
            style:
              fontSize: 24
              fontWeight: 'bold'
              backgroundColor: 'black'
              color: 'white'
            song


    UL 
      style:
        marginTop: 50
        listStyle: 'none'
        paddingLeft: 60

      for song in all
        LI null,

          A
            href: "/songs/#{song}"
            song



dom.SONG = -> 
  song = @props.song

  tasks = ['inventory', 'alignment', 'asides', 'aside editor', 'reactors', 'backchannels', 'layout', 'composition']

  loc = retrieve('location')

  parts = loc.path.split('/')

  song_config = retrieve("/song_config/#{song}")

  if parts.length < 4

    loc.path = loc.path + "/#{tasks[0]}"
    save loc
    return SPAN null

  else
    task = decodeURI parts[3]

  asty = 
    color: 'black'
    textDecoration: 'none'
    padding: '8px 14px'
    fontWeight: 700
    display: 'inline-block'

  DIV null,

    DIV 
      style:
        width: '100%'
        backgroundColor: '#68c4f5'
        padding: '0px 0px'


      SPAN
        style: 
          padding: '0 24px'
        song

      for avail in tasks

        A
          style: extend {}, asty,
            backgroundColor: if avail == task then '#368fbe'
            color: if avail == task then 'white' else 'black'
          href: [parts[0], parts[1], parts[2], avail].join('/')
          avail


      BUTTON
        onClick: =>
          editor = retrieve('json_editor')
          editor.active = !editor.active
          if editor.active
            editor.song = @props.song
          save editor
        'JSON'

    DIV
      style:
        backgroundColor: 'white'
        padding: '24px 18px'
        display: 'flex'


      if task == 'composition'
        COMPOSITION
          song: @props.song
      else if task == 'inventory'
        INVENTORY
          song: @props.song
      else if task == 'aside editor'
        ASIDE_EDITOR_LIST
          song: @props.song
      else if task == 'layout'
        LAYOUT
          song: @props.song
      else
        REACTION_LIST
          song: @props.song   
          task: task   

      if retrieve('json_editor').active and song_config.config
        DIV
          onMouseEnter: (evt) ->
            document.body.style.overflow = 'hidden'
          onMouseLeave: (evt) ->
            document.body.style.overflow = ''

          style: 
            resize: 'horizontal'
            overflowY: 'auto'
            flexGrow: 1
            # flexShrink: 1
            padding: '0 10px'
            minWidth: 450
            position: 'sticky'
            top: 0
            alignSelf: 'flex-start'

          JSONEditorTextArea
            json: song_config.config
            key: "#{JSON.stringify(song_config.config).length}" # update text area if it changes elsewhere
            onChange: (txt) => 
              @local.updated_json = JSON.parse(txt)
              save @local
            onSave: =>
              if @local.updated_json
                song_config.config = @local.updated_json
                save song_config
                @local.updated_json = null
          # JSONEditorWithSchema()




dom.LAYOUT = -> 
  song = @props.song
  song_config = retrieve("/song_config/#{song}")

  layout = song_config.layout
  return SPAN null unless layout

  el_width = (window.innerWidth * .95)
  grid_size = layout.grid_size
  adjustment = el_width / grid_size[0]
  el_height = adjustment * grid_size[1]

  assigned_seats = {}
  for k,v of layout.assignments
    assigned_seats[v] = k

  seat_size = layout.seat_size
  @local.selected ?= {}

  seats_selected = (k for k,v of @local.selected when v)
  num_selected = seats_selected.length

  base_size = layout.base_size
  base_location = layout.base_location

  @local.base_location ?= layout.base_location.slice()


  border_size = 1
  inner_border = 4

  DIV null,

    DIV null,
      "#{Object.keys(assigned_seats).length} seats assigned"


    DIV null,
      INPUT
        type: 'number'
        defaultValue: @local.base_location[0]
        onChange: (e) => 
          @local.base_location[0] = e.target.value
          console.log("x translated: #{@local.base_location[0] - layout.base_location[0]}")
          save @local
      INPUT
        type: 'number'
        defaultValue: @local.base_location[1]        
        onChange: (e) => 
          @local.base_location[1] = e.target.value
          console.log("y translated: #{@local.base_location[1] - layout.base_location[1]}")          
          save @local


      INPUT
        type: 'range'
        defaultValue: 60
        min: 0
        max: 240
        onChange: (ev) =>
          for vid in @getDOMNode().querySelectorAll('video')
            vid.currentTime = ev.target.value


    DIV 
      style: 
        position: 'relative'
        width: el_width
        height: el_height


      DIV 
        style:
          position: 'absolute'
          left: 0
          right: 0
          width: '100%'
          height: '100%'

        VIDEO 
          style:
            width: '100%'
            height: '100%'
          controls: false

          SOURCE  
            src: "/Media/#{song}/background.mp4"
            type: "video/mp4"

        # IMG 
        #   style:
        #     width: '100%'
        #     height: '100%'
        #   src: "/Media/#{song}/background.png"



      DIV
        style:
          position: 'absolute'
          left: @local.base_location[0] * adjustment
          top: @local.base_location[1] * adjustment

        VIDEO 
          width: base_size[0] * adjustment
          #height: base_size[1] * adjustment
          controls: false
          style:
            border: '3px solid white'
            borderRadius: 8

          SOURCE  
            src: "/Media/#{song}/#{song}.mp4"
            type: "video/mp4"

          SOURCE  
            src: "/Media/#{song}/#{song}.webm"
            type: "video/webm"



      for seat in layout.seats 

        selected = @local.selected[seat]
        click_seat = do (seat) => => 
          @local.selected[seat] =  !@local.selected[seat]
          save @local 
        DIV
          key: "#{seat}-#{assigned_seats[seat]}"
          style: 
            zIndex: if selected then 10 else 1
            position: 'absolute'
            left: (seat[0] - seat_size / 2) * adjustment - border_size
            top: (seat[1] - seat_size / 2) * adjustment - border_size

          DIV 
            style: 
              position: 'relative'
              width: seat_size * adjustment
              height: seat_size * adjustment
              #border: "#{inner_border}px solid transparent" 
              #borderColor: if selected then 'magenta'

              # maskImage: "url(\"/assets/hexagon.svg\")" #"radial-gradient(circle, white 100%, black 100%)"
              # maskSize: '100%'
              # boxShadow: if selected then '0 1px 2px magenta'
              background: if selected then "url(\"/assets/hexagon.svg\")"
              backgroundSize: if selected then 'contain'

            DIV 
              style: 
                backgroundColor: if seat not of assigned_seats then '#eaeaea' else '#ccc'

                width: seat_size * adjustment - inner_border * 2
                height: seat_size * adjustment - inner_border * 2
                maskImage: "url(\"/assets/hexagon.svg\")" #"radial-gradient(circle, white 100%, black 100%)"
                maskSize: '100%'
                position: 'relative'
                left: inner_border
                top: inner_border
                marginRight: inner_border
                marginBottom: inner_border

              onClick: if seat not of assigned_seats then click_seat

              if seat of assigned_seats
                src = assigned_seats[seat]
                parts = src.split('/')
                reaction_file_prefix = parts[parts.length - 1].split('-CROSS-')[0]
                gaze = get_gaze_direction(song, reaction_file_prefix)
                if gaze
                  [sel_hori, sel_vert] = gaze
                  flipx = (seat[0] < grid_size[0] / 2 && sel_hori == 'left' ) || (seat[0] > grid_size[0] / 2 && sel_hori == 'right' )
                else
                  flipx = false

                VIDEO 
                  "data-tooltip": assigned_seats[seat]
                  width: '100%'
                  height: '100%'                
                  style:
                    # borderRadius: "50%"
                    objectFit: "cover"
                    transform: if flipx then "scaleX(-1)"

                  controls: false
                  onClick: click_seat
                  onDoubleClick: (evt) ->
                    saveMaskedVideoFrame(evt.target)

                  SOURCE  
                    src: '/' + assigned_seats[seat]
                    type: "video/mp4"

          if selected && num_selected == 2
            DIV
              style:
                position: 'absolute'
                top: -20
                left: "calc(50%-#{seat_size / 2 * adjustment}px)"

              BUTTON
                onClick: =>
                  position_1 = seats_selected[0]
                  assignment_1 = assigned_seats[position_1]

                  position_2 = seats_selected[1]
                  assignment_2 = assigned_seats[position_2]

                  if assignment_1
                    layout.assignments[assignment_1] = (parseFloat(f) for f in position_2.split(','))
                  if assignment_2
                    layout.assignments[assignment_2] = (parseFloat(f) for f in position_1.split(','))

                  console.log("SWAP", {assignment_1, position_1}, {assignment_2, position_2})
                  save song_config

                  delete @local.selected[position_1]
                  delete @local.selected[position_2]
                  save @local

                'Swap'



dom.LAYOUT.refresh = ->
  return if !@getDOMNode()
  @loaded_videos ?= {}
  func =  (ev) -> 
      ev.target.currentTime = 60
      ev.target.removeEventListener('canplaythrough', func)

  for vid in @getDOMNode().querySelectorAll('video')
    if vid.firstChild.src not of @loaded_videos
      @loaded_videos[vid.firstChild.src] = true

      if vid.readyState > 1
        vid.currentTime = 60
      else 
        vid.addEventListener 'canplaythrough', func



saveMaskedVideoFrame = (video) ->
    canvas = document.createElement('canvas')
    canvas.width = w = video.videoWidth
    canvas.height = h = video.videoHeight
    s = Math.min(w, h)

    ctx = canvas.getContext('2d')

    # Define hexagon path
    hexagonPath = (ctx, x, y, radius) ->
        ctx.beginPath()
        for i in [0..6]
          ctx.lineTo(x + 1.25 * radius * Math.cos((Math.PI / 3) * i - Math.PI / 2),
                     y + radius * Math.sin((Math.PI / 3) * i - Math.PI / 2))
        ctx.clip()


    # Clip to hexagon
    hexagonPath(ctx, w / 2, h / 2, s / 2)

    # Draw video frame
    ctx.drawImage(video, 0, 0, w, h)

    # To save the canvas image including the hexagonal transparency,
    # convert the canvas to a data URL and then to an image file if needed.
    # imageDataUrl = canvas.toDataURL('image/png')

    # # For demonstration, create an image element to display the result
    # img = document.createElement('img')
    # img.src = imageDataUrl
    # img.width=img.height=104
    # document.body.appendChild(img) # Append or handle the image data URL as needed


    canvas.toBlob (blob) ->
        console.log("Converting to blob")
        item = new ClipboardItem({ "image/png": blob })
        console.log('created new clipboard item')
        navigator.clipboard.write([item]).then ->
            console.log("Hexagonal image copied to clipboard.")
        , (error) ->
            console.error("Error copying image to clipboard: ", error)

        console.log('wrote to clipboard')





dom.COMPOSITION = ->
  song = @props.song


  DIV null,

    VIDEO
      src: "/media/#{song}/#{song} (composition).mp4"
      controls: true
      playsinline: true
      width: 600




dom.LIST_TOOL_BAR = -> 
  local = retrieve(@props.local)
  task = @props.task
  registered_media = @props.registered_media

  DIV
    className: 'process-actions'
    style: 
      position: 'sticky'
      top: 0
      zIndex: 9
      display: 'flex'
      alignSelf: 'flex-start'
      backgroundColor: 'white'


    if task != 'aside editor'
      BUTTON 
        key: "play_all #{local.play_all}"
        onClick: => 
          local.play_all = !local.play_all
          song_playing = false

          selected = @getDOMNode().parentNode.querySelector('[data-is-selected=true]')

          for k,v of registered_media
            console.log(k,v)
            is_song = v.is_song
            mute = is_song && song_playing
            is_audible = v.play(local.play_all, mute, !selected)
            song_playing ||= is_song && is_audible
          save local

        if local.play_all
          'Pause all'
        else
          'Play selected'

    if task != 'aside editor'
      BUTTON 
        key: "select_all #{local.all_selected}"
        onClick: => 
          local.all_selected = !local.all_selected
          if local.all_selected
            local.force_selection = true
          else
            local.force_deselection = true
          save local

        if !local.all_selected
          'Select all'
        else
          'Unselect all'

    BUTTON 
      key: 'hide_unselected'
      onClick: => 
        local.hide_unselected = !local.hide_unselected
        save local
      
      if local.hide_unselected
        'Show all'
      else
        'Hide unselected'


    if task == 'alignment'
      BUTTON 
        key: 'show reactions'
        onClick: => 
          local.show_reactions = !local.show_reactions
          save local
        
        if local.show_reactions
          'Hide reactions'
        else
          'Show reactions'

    if task == 'alignment'
      BUTTON 
        key: 'show paintings'
        onClick: => 
          local.show_paintings = !local.show_paintings
          save local
        
        if local.show_paintings
          'Hide paintings'
        else
          'Show paintings'

    DIV null,

      BUTTON 
        onClick: => 
          local.page -= 1
          if local.page < 0
            local.page = 0
          save local
        '<'

      BUTTON
        onClick: =>
          local.page += 1
          save local
        '>'

    INPUT 
      key: 'filter'
      type: 'text'
      defaultValue: local.filter_reactions or "" 
      onChange: (e) => 
        local.filter_reactions = e.target.value
        local.page = 0
        save local

    BUTTON
      onClick: (e) => 
        local.filter_to_starred = !local.filter_to_starred
        local.page = 0
        save local

      I
        className: 'glyphicon glyphicon-star'

    BUTTON
      onClick: (e) => 
        local.sort_by_alignment_score = !local.sort_by_alignment_score
        local.page = 0
        save local

      I
        className: 'glyphicon glyphicon-exclamation-sign'

    BUTTON
      onClick: (e) => 
        local.filter_to_alignment_incomplete = !local.filter_to_alignment_incomplete
        local.page = 0
        save local

      I
        className: 'glyphicon glyphicon-ban-circle'



dom.REACTION_LIST = -> 
  song = @props.song
  manifest = retrieve("/manifest/#{song}")

  channels = retrieve('/channels').channels
  return DIV null if !channels

  if !manifest.manifest?.reactions
    return SPAN null

  all_reactions = Object.values(manifest.manifest.reactions)

  downloaded_reactions = []
  for o in all_reactions
    if o.download
      if !@local.filter_reactions || (o.reactor.toLowerCase().indexOf(@local.filter_reactions.toLowerCase()) > -1)
        metadata = retrieve("/reaction_metadata/#{song}/#{o.id}")
        if metadata.alignment
          downloaded_reactions.push o

  marked_compare = (a,b) -> 
    if a.marked == b.marked
      return alphabetical_compare(a.reactor,b.reactor)
    else
      if a.marked
        return -1
      else
        return 1



  if @local.filter_to_starred
    downloaded_reactions = (d for d in downloaded_reactions when d.marked)    
      # downloaded_reactions.sort( (a,b) -> marked_compare(a,b) )
    # else

  if @local.filter_to_alignment_incomplete
    downloaded_reactions = (d for d in downloaded_reactions when !d.alignment_done) 

  if @local.sort_by_alignment_score
    downloaded_reactions.sort( (a,b) -> retrieve("/reaction_metadata/#{song}/#{a.id}").alignment_score - retrieve("/reaction_metadata/#{song}/#{b.id}").alignment_score     )
  else
    downloaded_reactions.sort( (a,b) -> alphabetical_compare(a.reactor,b.reactor)     )

  task = @props.task

  @registered_media ?= {}

  if @local.force_selection
    select_all = true
    @local.force_selection = false
  if @local.force_deselection
    deselect_all = true
    @local.force_deselection = false

  @local.page ?= 0

  if task == 'backchannels'
    @local.per_page = 50
  else
    @local.per_page = 10


  DIV null,

    LIST_TOOL_BAR 
      local: @local
      task: task
      registered_media: @registered_media

        
    UL 
      style:
        listStyle: 'none'
        paddingLeft: 24

      for reaction, i in downloaded_reactions
        metadata = retrieve("/reaction_metadata/#{song}/#{reaction.id}")
        retrieve("/reaction/#{reaction.id}") # subscribe to updates to reaction
        if metadata.alignment and i >= @local.per_page * @local.page and i < @local.per_page * (@local.page + 1)
          REACTION_ITEM
            key: "#{reaction.id}"
            song: song
            reaction: reaction
            task: task
            registered_media: @registered_media
            hide_unselected: @local.hide_unselected
            show_reaction: @local.show_reactions
            show_painting: @local.show_paintings
            force_selection: select_all
            force_deselection: deselect_all


dom.REACTION_ITEM = ->

  song = @props.song
  reaction = @props.reaction

  key = "/reaction/#{reaction.id}"
  retrieve(key)

  metadata = retrieve("/reaction_metadata/#{song}/#{reaction.id}")
  song_config = retrieve("/song_config/#{@props.song}")
  config = song_config.config

  reaction_file_prefix = reaction.file_prefix or reaction.reactor
  retrieve("/action/#{reaction.id}") # subscribe to actions on this reaction

  task = @props.task
  hide_unselected = @props.hide_unselected

  song_dir = ["/media", song].join('/')
  meta_dir = [song_dir, 'bounded'].join('/')
  reactions_dir = [song_dir, 'reactions'].join('/')


  aligned_video = [meta_dir, "#{reaction_file_prefix}-CROSS-EXPANDER" ].join('/')
  reaction_video = [reactions_dir, "#{reaction_file_prefix}" ].join('/')
  song_video = [song_dir, song].join('/')

  alignment_painting = [meta_dir, "#{reaction_file_prefix}-painting-3.png" ].join('/')

  isolated_backchannel = [meta_dir, "#{reaction_file_prefix}-isolated_backchannel.wav" ].join('/')

  if @props.force_selection
    @local.selected = true
  else if @props.force_deselection
    @local.selected = false

  if task == 'alignment'
    if @props.show_reaction
      vids = [song_video, aligned_video, reaction_video]      
    else
      vids = [song_video, aligned_video]
  else if task == 'backchannels'
    vids = [isolated_backchannel]
  else if task == 'asides'
    vids = [song_video, reaction_video, reaction_video]
  else if task == 'aside editor'
    vids = [reaction_video]
  else if task == 'reactors'
    vids = [aligned_video]
    for v in metadata.reactors or [] 
      vids.push [meta_dir, v ].join('/')
  else
    vids = [song_video, aligned_video, reaction_video]


  retrieve("/action/#{reaction.id}") # subscribe to actions on this reaction

  if hide_unselected and !@local.selected
    return LI null


  if @local.clicked_at
    if config.fake_reactor_position?[reaction_file_prefix] && @local.clicked_at in config.fake_reactor_position[reaction_file_prefix]
      @local.clicked_at = null

  LI 
    key: "#{reaction.id} #{@local.key} #{task}"
    style: 
      backgroundColor: if @local.selected then '#dadada' else 'transparent'
      padding: '4px 12px'
      margin: '4px 0'
      borderRadius: 16
      display: if hide_unselected and !@local.selected then 'none' else 'flex'
    onDoubleClick: =>
      @local.selected = !@local.selected
      save @local

    DIV 
      style: 
        display: 'flex'
        width: 250
        overflow: 'hidden'
        alignItems: 'center'
        # height: 200

      # if task == 'alignment'
      #   SPAN 
      #     style: 
      #       color: '#555'
      #       fontSize: 12

      #     "#{Math.round(metadata.alignment_score)}"


      DIV null,

        BUTTON 
          title: 'Sync media for this reaction'
          style:
            backgroundColor: 'transparent'
            border: 'none'
            outline: 'none'
            cursor: 'pointer'
            opacity: if @local.disable_sync then .5 else 1

          onClick: => 
            @local.disable_sync = !@local.disable_sync
            save @local
            e.stopPropagation()

          I 
            className: 'glyphicon glyphicon-link'




        if task in ['alignment', 'reactors', 'backchannels', 'asides']
          BUTTON
            style:
              cursor: 'pointer'
              border: 'none'
              backgroundColor: 'transparent'
              outline: 'none'
              margin: '0 4px'
            onClick: (e) =>
              if task == 'alignment'
                if confirm('Reset will delete all processed metadata for this reaction. You sure?')
                  action = 
                    key: "/action/#{reaction.id}"
                    action: 'delete'
                    scope: 'alignment'
                    reaction_id: reaction.id
                    song: song
                  save action
              else if task == 'reactors'
                if confirm('Reset will delete all cropped reactor videos. You sure?')
                  action = 
                    key: "/action/#{reaction.id}"
                    action: 'delete'
                    scope: 'cropped reactors'
                    reaction_id: reaction.id
                    song: song

                  if confirm('Redo coarse face tracking too? Cancel means just redo fine-grained tracking.')
                    action.scope = 'cropped reactors including coarse'

                  save action
              else if task == 'backchannels'
                if confirm('Reset will delete isolated backchannel files. You sure?')
                  action = 
                    key: "/action/#{reaction.id}"
                    action: 'delete'
                    scope: 'isolated backchannel'
                    reaction_id: reaction.id
                    song: song
                  save action
              else if task == 'asides'
                if confirm('Reset will delete all asides files for this reaction. You sure?')
                  action = 
                    key: "/action/#{reaction.id}"
                    action: 'delete'
                    scope: 'asides'
                    reaction_id: reaction.id
                    song: song
                  save action

              e.stopPropagation()

            I
              className: 'glyphicon glyphicon-refresh'


        BUTTON 
          title: 'Mark reaction'
          style:
            backgroundColor: 'transparent'
            border: 'none'
            outline: 'none'
            cursor: 'pointer'
            opacity: if !reaction.marked then .5 else 1

          onClick: (e) => 
            reaction.marked = !reaction.marked
            save {
              key: "/reaction/#{reaction.id}",
              reaction: reaction, 
              song: song
            }
            e.stopPropagation()

          I 
            className: 'glyphicon glyphicon-star'

        BUTTON 
          title: 'Alignment Complete'
          style:
            backgroundColor: 'transparent'
            border: 'none'
            outline: 'none'
            cursor: 'pointer'
            opacity: if !reaction.alignment_done then .5 else 1

          onClick: (e) => 
            reaction.alignment_done = !reaction.alignment_done
            save {
              key: "/reaction/#{reaction.id}",
              reaction: reaction, 
              song: song
            }
            e.stopPropagation()

          I 
            className: 'glyphicon glyphicon-ok'

        




      BUTTON
        onClick: => 
          navigator.clipboard.writeText(reaction.reactor)
          e.stopPropagation()


        style: 
          border: 'none'
          backgroundColor: 'transparent'
          outline: 'none'
          cursor: 'pointer'
          display: 'block'
          padding: "8px 10px"
          textAlign: 'left'

        reaction.reactor

          






    DIV 
      style: 
        display: 'flex'

      for video, idx in vids
        DIV
          key: "#{idx} #{reaction_video == video} #{song_video == video} #{aligned_video == video}"
          style: 
            marginRight: 12


          if task == 'backchannels'
            SYNCHRONIZED_AUDIO
              keep_synced: !@local.disable_sync
              selected: @local.selected
              audio: isolated_backchannel
              alignment_data: metadata.alignment.best_path
              is_reaction: false
              is_song: false
              registered_media: @props.registered_media   
              task: task  
              song: song   
              reaction_file_prefix: reaction_file_prefix  


          else if task == 'aside editor'
            SYNCHRONIZED_AUDIO
              keep_synced: !@local.disable_sync
              selected: @local.selected
              audio: isolated_backchannel
              alignment_data: metadata.alignment.best_path
              is_reaction: false
              is_song: false
              registered_media: @props.registered_media   
              task: task  
              song: song   
              reaction_file_prefix: reaction_file_prefix  


          else
            SYNCHRONIZED_VIDEO
              keep_synced: !@local.disable_sync
              selected: @local.selected
              video: video
              alignment_data: metadata.alignment.best_path
              is_reaction: reaction_video == video
              is_song: song_video == video
              registered_media: @props.registered_media
              soundfile: if aligned_video == video then isolated_backchannel
              task: task
              song: song
              reaction_file_prefix: reaction_file_prefix
              onInitialize: do(idx) => (key, time_key) =>
                @local["video-#{idx}-time-key"] = key
              onTimeClicked: do(idx) => (e, time) => 
                @local["video-#{idx}-time"] = time
                save @local

              onVideoClicked: (e) => 
                if task == 'reactors'
                  x = event.offsetX
                  y = event.offsetY
                  w = e.target.clientWidth
                  h = e.target.clientHeight

                  @local.clicked_at = [x / w, y / h]
                  save @local

          if task == 'alignment' && @props.show_reaction && idx == 2
            DIV null, 
              BUTTON 
                title: 'Mark start of reaction'
                style:
                  backgroundColor: 'transparent'
                  border: 'none'
                  outline: 'none'
                  cursor: 'pointer'

                onClick: (e) => 
                  reaction_time = retrieve("time-#{@local["video-2-time-key"]}").time

                  if confirm("Mark #{reaction_time}s as the start of the reaction?")
                    song_config.config.start_reaction_search_at ?= {}
                    song_config.config.start_reaction_search_at[reaction.reactor] = reaction_time
                    save(song_config)

                  e.stopPropagation()

                I 
                  className: 'glyphicon glyphicon-triangle-right'

              BUTTON 
                title: 'Mark end of reaction'
                style:
                  backgroundColor: 'transparent'
                  border: 'none'
                  outline: 'none'
                  cursor: 'pointer'

                onClick: (e) => 
                  reaction_time = retrieve("time-#{@local["video-2-time-key"]}").time

                  if confirm("Mark #{reaction_time}s as the start of the reaction?")
                    song_config.config.end_reaction_search_at ?= {}
                    song_config.config.end_reaction_search_at[reaction.reactor] = reaction_time
                    save(song_config)

                  e.stopPropagation()

                I 
                  className: 'glyphicon glyphicon-triangle-left'

              BUTTON 
                title: 'Manual bind'
                style:
                  backgroundColor: 'transparent'
                  border: 'none'
                  outline: 'none'
                  cursor: 'pointer'

                onClick: (e) => 
                  song_time = retrieve("time-#{@local["video-0-time-key"]}").time
                  reaction_time = retrieve("time-#{@local["video-2-time-key"]}").time

                  if confirm("Bind #{reaction_time}s of the reaction to #{song_time}s of the song?")
                    song_config.config.manual_bounds ?= {}
                    song_config.config.manual_bounds[reaction.reactor] ?= []
                    song_config.config.manual_bounds[reaction.reactor].push( [song_time, reaction_time]  )
                    save(song_config)

                  e.stopPropagation()

                I 
                  className: 'glyphicon glyphicon-resize-small'


    if task == 'asides'

      insert_at = @local['video-0-time']
      start = @local['video-1-time']
      end = @local['video-2-time']

      DIV 
        style:
          paddingTop: 30

        if (config.asides?[reaction_file_prefix] or []).length > 0
          DIV null,
            LABEL null, 
              'Asides: '
            for aside, aside_idx in (config.asides?[reaction_file_prefix] or [])

              BUTTON 
                style: 
                  backgroundColor: if aside_idx == @local.editing_aside?[1] then '#ccc'
                onClick: do (aside, aside_idx) => => 
                  @local.editing_aside = [aside, aside_idx]
                  delete @local['video-0-time']
                  delete @local['video-1-time']
                  delete @local['video-2-time']
                  save @local

                "#{aside_idx}"

        if @local.editing_aside
          EDIT_ASIDE
            key: "aside-#{aside_idx}-#{@local.editing_aside}"
            song: @props.song
            aside: @local.editing_aside
            fresh: false
            time_state_key: @local.key
            registered_media: @props.registered_media 
            reaction_file_prefix: reaction_file_prefix
            on_cancel: =>
              delete @local.editing_aside
              delete @local['video-0-time']
              delete @local['video-1-time']
              delete @local['video-2-time']
              save @local
        
        else if insert_at? && start?
          if !end?
            @local['video-2-time'] = start
            save @local
            @props.registered_media[@local['video-2-time-key']].set_time(start)



          EDIT_ASIDE
            song: @props.song
            fresh: true
            time_state_key: @local.key
            registered_media: @props.registered_media
            reaction_file_prefix: reaction_file_prefix
            on_save: =>
              delete @local['video-0-time']
              delete @local['video-1-time']
              delete @local['video-2-time']
              save @local

    if task == 'alignment' and @props.show_painting
      A
        href: alignment_painting
        target: '_blank'
        style: 
            display: 'inline-block'

        IMG
          style: 
            height: 240
          src: alignment_painting



    if task == 'reactors' and (metadata.reactors or []).length > 0
      REACTOR_TASKS
        song: song
        reaction: reaction
        clicked_at: @local.clicked_at



get_gaze_direction = (song, reaction_file_prefix, reaction_id) ->
  manifest = retrieve("/manifest/#{song}")

  if !manifest.manifest?.reactions
    return null

  all_reactions = Object.values(manifest.manifest.reactions)

  if !reaction_id
    reaction_id = null
    for r in all_reactions
      if (r.file_prefix or r.reactor) == reaction_file_prefix
        reaction_id = r.id
        break

  if !reaction_id
    console.log("COULD NOT FIND REACTION ID FOR #{reaction_file_prefix}")
    return null

  metadata = retrieve("/reaction_metadata/#{song}/#{reaction_id}")
  song_config = retrieve("/song_config/#{song}")
  config = song_config.config

  retrieve("/action/#{reaction_id}") # subscribe to actions on this reaction

  fname = metadata.reactors[0]
  parts = fname.split('-')
  vert = parts[parts.length - 1]
  hori = parts[parts.length - 2]

  override = config?.face_orientation?[reaction_file_prefix]

  sel_hori = override?[0] or hori
  sel_vert = override?[1] or vert

  return [sel_hori, sel_vert]



dom.REACTOR_TASKS = ->
  song = @props.song
  reaction = @props.reaction

  metadata = retrieve("/reaction_metadata/#{song}/#{reaction.id}")
  song_config = retrieve("/song_config/#{@props.song}")
  config = song_config.config

  reaction_file_prefix = reaction.file_prefix or reaction.reactor
  retrieve("/action/#{reaction.id}") # subscribe to actions on this reaction

  [sel_hori, sel_vert] = get_gaze_direction(song, reaction_file_prefix, reaction.id)

  num_reactors = config?.multiple_reactors?[reaction_file_prefix] or metadata.reactors.length

  featured = 'reaction_file_prefix' in (config?.featured or [])

  priority = config?.priority?[reaction_file_prefix] or (if featured then 75 else 50)


  DIV 
    style:
      fontSize: 16
      display: 'flex'
      flexDirection: 'column'

    DIV
      style: 
        display: 'flex'
      DIV null,

        SELECT 
          style: 
            fontSize: 18
          value: sel_hori
          onChange: (e) => 
            config.face_orientation ?= {}
            config.face_orientation[reaction_file_prefix] = [e.target.value, sel_vert]
            save song_config

          for val in ['left', 'right', 'center']
            OPTION 
              value: val
              style: {}
              val

      DIV null,
        SELECT  
          style: 
            fontSize: 18
          value: sel_vert
          onChange: (e) => 
            config.face_orientation ?= {}
            config.face_orientation[reaction_file_prefix] = [sel_hori, e.target.value]
            save song_config

          for val in ['up', 'down', 'middle']
            OPTION 
              value: val
              style: {}
              val
      
    DIV null,

      INPUT
        type: 'range'
        min: 0
        max: 100
        value: priority
        onChange: (e) =>
          config.priority ?= {} 
          config?.priority[reaction_file_prefix] = parseInt(e.target.value)
          save song_config


    if @props.clicked_at
      DIV null,
        SPAN null,
          "#{Math.round(@props.clicked_at[0] * 100)}% / #{Math.round(@props.clicked_at[1] * 100)}%"
        BUTTON 
          onClick: => 
            config.fake_reactor_position ?= {}
            config.fake_reactor_position[reaction_file_prefix] ?= []
            config.fake_reactor_position[reaction_file_prefix].push @props.clicked_at
            save song_config

          'Exclude as false positive reactor'

    SELECT
      style: 
        fontSize: 18
      value: num_reactors
      onChange: (e) =>
        config.multiple_reactors ?= {}
        config.multiple_reactors[reaction_file_prefix] = parseInt(e.target.value)
        save song_config

      for val in ['1','2','3','4','5']
        OPTION 
          value: val
          style: {}
          val


dom.SYNCHRONIZED_VIDEO = ->

  video = @props.video
  alignment_data = @props.alignment_data
  is_reaction = @props.is_reaction



  DIV null,

    VIDEO
      width: 320
      height: 240
      controls: true
      ref: 'video'
      'data-media': video
      'data-receive-viewport-visibility-updates': 2
      "data-component": @local.key
      "data-is-song": @props.is_song
      "data-is-reaction": @props.is_reaction
      "data-is-selected": @props.selected
      "data-keep-synced": @props.keep_synced

      SOURCE
        src: video + '.mp4'
        type: "video/mp4"

      SOURCE
        src: video + '.webm'
        type: "video/webm"


      onClick: (e) =>
        @props.onVideoClicked?(e)

    if is_reaction
      BEST_PATH_BAR
        alignment_data: alignment_data
        video: @props.video
    else
      DIV 
        style:
          height: 10
          width: '100%'

    if false and !is_reaction and !@props.is_song
      DIV
        ref: 'wavesurfer'
        style:
          height: 18
          width: 'calc(100% - 34px)'
          margin: '0 17px'



    TIME_DISPLAY
      time_state: "time-#{@local.key}"
      onTimeClicked: @props.onTimeClicked
      onInitialize: @props.onInitialize
      parent_key: @local.key


dom.BEST_PATH_BAR = ->
  alignment_data = @props.alignment_data
  video = @props.video

  duration = @local.duration or alignment_data[alignment_data.length - 1][1] / sample_rate
  segments = []
  
  last_reaction_end = 0

  for segment in alignment_data

    reaction_start = segment[0] / sample_rate
    reaction_end = segment[1] / sample_rate
    base_start = segment[2] / sample_rate
    base_end = segment[3] / sample_rate


    if last_reaction_end < reaction_start - 1
      segments.push 
        type: 'speaking'
        length: reaction_start - last_reaction_end
        start: last_reaction_end
        end: reaction_start


    segments.push 
      type: 'backchannel'
      length: reaction_end - reaction_start
      start: reaction_start
      end: reaction_end

    last_reaction_end = reaction_end

  if last_reaction_end < duration - 1
    segments.push 
      type: 'speaking'
      length: duration - last_reaction_end
      start: last_reaction_end
      end: duration



  DIV 
    style: 
      width: 'calc(100% - 34px)'
      margin: '0 17px'

    STYLE """
        .path_bar_segment {
          opacity: .7;
        }
        .path_bar_segment:hover {
          opacity: 1;
        }

      """

    for segment in segments
      BUTTON
        className: 'path_bar_segment'
        style:
          display: 'inline-block'
          height: 10
          cursor: 'pointer'
          width: "#{segment.length / duration * 100}%"
          backgroundColor: if segment.type == 'speaking' then '#ddb2b5' else '#ddd'
          outline: 'none'
          border: 'none'
          padding: 0
        onClick: do(segment) => (ev) =>
          video = document.querySelector("[data-media=\"#{@props.video}\"]")
          video.currentTime = segment.start



dom.BEST_PATH_BAR.refresh = ->
  if !@initialized
    video = document.querySelector("[data-media=\"#{@props.video}\"]")
    if video
      @initialized = true
      if video.duration and !isNaN(video.duration)
        @local.duration = video.duration
        save @local
      else
        video.addEventListener 'canplay', => 
          @local.duration = video.duration
          save @local


dom.TIME_DISPLAY = ->
  time = retrieve(@props.time_state).time or 0

  DIV
    style:
      color: 'black'
      cursor: 'pointer'
      display: 'flex'
      alignItems: 'center'

    onClick: (ev) =>
      active = retrieve('active_number')
      active.number = time
      save active

      navigator.clipboard.writeText(active.number)

      @props.onTimeClicked?(ev, time)

    "#{time}"

dom.TIME_DISPLAY.up = ->
  @props.onInitialize?(@props.parent_key, @props.time_state)





# Translates a time from reaction time to base time.
# alignment_data is a list of segment alignment tuples in the format 
#      (reaction_start, reaction_end, base_start, base_end, ...)
#   ...where we only need the first four values. 
#   reaction_start, reaction_end give the start and time of the segment
#   in reaction time. 
#   base_start, base_end give the start and end of the corresponding 
#   segment in the base time.
# Each of these values is an audio sample rate of 44100, so to get to 
# seconds, divide the value by 44100. 
# These values can be used to construct 
sample_rate = 44100
get_base_time = (reaction_vid_time, alignment_data) ->


  current_base = 0
  current_reaction = 0

  for segment in alignment_data
    reaction_start = segment[0] / sample_rate
    reaction_end = segment[1] / sample_rate
    base_start = segment[2] / sample_rate
    base_end = segment[3] / sample_rate

    seg_length = reaction_end - reaction_start


    if reaction_vid_time <= reaction_start + seg_length
      if reaction_vid_time >= reaction_start   # a part of the reaction in the base video
        btime = current_base + (reaction_vid_time - reaction_start)
        # console.log("FOUND BASE TIME #{btime} FROM #{reaction_vid_time}")
        return btime
      else # could not find a match, return the last base end
        return current_base

    else
      current_base = base_end
      current_reaction += seg_length
      last_reaction_end = reaction_end

  console.log("CANT FIND BASE TIME FROM #{reaction_vid_time}")
  return null



get_reaction_time = (base_time, alignment_data) ->

  if base_time == 0
    return alignment_data[0][0] / sample_rate

  current_base = 0

  for segment in alignment_data
    reaction_start = segment[0] / sample_rate
    reaction_end = segment[1] / sample_rate
    base_start = segment[2] / sample_rate
    base_end = segment[3] / sample_rate

    console.assert( Math.abs(  Math.abs(reaction_start - reaction_end) - Math.abs(base_start - base_end) ) < .0001    )
    seg_length = reaction_end - reaction_start

    if base_time <= current_base + seg_length
      rtime = reaction_start + (base_time - base_start)
      # console.log("FOUND REACTION TIME #{rtime} FROM #{base_time}")
      return rtime

    else
      current_base = base_end

  console.log("CANT FIND REACTION TIME FROM #{base_time}")

  return null



dom.SYNCHRONIZED_VIDEO.down = ->
  delete @props.registered_media[@local.key]

dom.SYNCHRONIZED_VIDEO.refresh = ->

  vid = @refs.video.getDOMNode()
  is_reaction = @props.is_reaction
  alignment_data = @props.alignment_data



  if !@initialized
    @initialized = true


    @props.registered_media[@local.key] = 
      is_song: @props.is_song
      play: (play, mute, force) =>

        if !play
          vid.pause()
          vid.muted = false
        else if @props.selected || force
          vid.play()
          vid.muted = mute
          return mute
        return false 

      set_time: (base_time) => 
        @ignore = true
        vid.currentTime = base_time
        setTimeout =>
          @ignore = false
        , 1500

      synchronize: (base_time) =>
        if !@props.keep_synced
          return

        if is_reaction
          currentTime = get_base_time(vid.currentTime, alignment_data)

          if Math.abs(base_time - currentTime) > .00001
            translated_time = get_reaction_time(base_time, alignment_data)

            @ignore_seek = translated_time
            vid.currentTime = translated_time
        else
          if base_time != vid.currentTime
            @ignore_seek = base_time
            vid.currentTime = base_time

    if is_reaction
      vid.currentTime = get_reaction_time(0, alignment_data) # initialize to beginning of base video in the reaction

    handle_seek = (ev) =>
      if @ignore
        return

      ts = ev.target.currentTime
      if Math.abs(@ignore_seek - ts) < .00001
        return

      if is_reaction
        ts = get_base_time(ts, alignment_data)

      syncers = Object.keys(@props.registered_media)
      syncers.sort( (a,b) -> (if bus.cache[b]?.in_viewport then 1 else 0) - (if bus.cache[a]?.in_viewport then 1 else 0)  )
      for k in syncers 
        if k != @local.key
          @props.registered_media[k].synchronize(ts)

    vid.addEventListener 'seeked', handle_seek


    handle_time = (ev) =>
      time_state = retrieve("time-#{@local.key}")
      time_state.time = ev.target.currentTime
      save time_state

    vid.addEventListener 'timeupdate', handle_time

    if @props.is_song
      vid.volume = .2


  if @refs.wavesurfer && @refs.wavesurfer && !@wavesurfer_added
    @wavesurfer_added = true
    wavesurfer = WaveSurfer.create
      container: @refs.wavesurfer.getDOMNode()
      waveColor: 'rgb(200, 0, 200)'
      progressColor: 'rgb(100, 0, 100)'
      url: encodeURI(@props.soundfile)
      height: 'auto'
      normalize: true
      # url: encodeURI("http://#{retrieve('location').host}#{@props.soundfile}")


    wavesurfer.on 'click', =>
      if !wavesurfer.isPlaying()
        wavesurfer.play()

    wavesurfer.on 'dblclick', =>
      if wavesurfer.isPlaying()
        wavesurfer.pause()









dom.SYNCHRONIZED_AUDIO = ->

  audio = @props.audio
  alignment_data = @props.alignment_data

  DIV   
    style: 
      display: 'flex'
      style:
        'align-items': 'center'

    DIV
      'data-media': audio
      'data-receive-viewport-visibility-updates': 2
      "data-component": @local.key
      "data-is-song": @props.is_song
      "data-is-reaction": @props.is_reaction
      "data-is-selected": @props.selected
      "data-keep-synced": @props.keep_synced

      ref: 'wavesurfer'
      style:
        height: 32
        width: 600
        # width: 'calc(100% - 34px)'
        margin: '0 17px'


    TIME_DISPLAY
      time_state: "time-#{@local.key}"


dom.SYNCHRONIZED_AUDIO.down = ->
  delete @props.registered_media[@local.key]

dom.SYNCHRONIZED_AUDIO.refresh = ->
  if !@refs.wavesurfer?
    return

  if !@initialized
    synchronize_wavesurfer = (base_time) =>

      if !@props.keep_synced
        return

      if base_time != wavesurfer.getCurrentTime
        @ignore_seek = base_time
        wavesurfer.setTime base_time

    wavesurfer = WaveSurfer.create
      container: @refs.wavesurfer.getDOMNode()
      waveColor: 'rgb(200, 0, 200)'
      progressColor: 'rgb(100, 0, 100)'
      url: encodeURI(@props.audio)
      height: 'auto'
      normalize: true


    # wavesurfer.on 'click', =>
    #   if !wavesurfer.isPlaying()
    #     wavesurfer.play()

    # wavesurfer.on 'dblclick', =>
    #   if wavesurfer.isPlaying()
    #     wavesurfer.pause()


    @initialized = true

    @props.registered_media[@local.key] = 
      is_song: @props.is_song
      play: (play, mute, force) =>
        if play && (force || @props.selected)
          wavesurfer.play()
          wavesurfer.setMuted(mute)
        else if !play
          wavesurfer.pause()
          wavesurfer.setMuted(false)
      synchronize: synchronize_wavesurfer

    handle_seek = (ts) =>

      if Math.abs(@ignore_seek - ts) < .00001
        return

      syncers = Object.keys(@props.registered_media)
      syncers.sort( (a,b) -> (if bus.cache[b]?.in_viewport then 1 else 0) - (if bus.cache[a]?.in_viewport then 1 else 0)  )
      for k in syncers 
        if k != @local.key
          @props.registered_media[k].synchronize(ts)

    wavesurfer.on 'seeking', handle_seek


    handle_time = (currentTime) =>
      time_state = retrieve("time-#{@local.key}")
      time_state.time = currentTime
      save time_state

    wavesurfer.on 'timeupdate', handle_time

    if @props.is_song
      wavesurfer.setVolume = .2


