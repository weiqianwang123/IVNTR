;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Clean Table Real Domain
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain clean-table-real)
  (:requirements :strips :typing :adl :derived-predicates)
  (:types robot toy wiper box table)
  (:predicates 
    (toy_on_table ?t - toy)
    (handempty ?r - robot)
    (holdingToy ?r - robot ?t - toy)
    (toy_in_box ?t - toy ?b - box)
    (wiper_in_box ?w - wiper ?b - box)
    (wiper_on_table ?w - wiper)
    (holdingWiper ?r - robot ?w - wiper)
    (box_at_center ?b - box)
    (box_at_side ?b - box)
    (table_clean ?tb - table)
    (goalAchieved)
  )

  ;; Derived predicate: true when no toys are on the table
  (:derived (No_toy_at_table)
    (forall (?t - toy) (not (toy_on_table ?t))))

  (:action PickToyFromTable
    :parameters (?r - robot ?t - toy)
    :precondition (and (toy_on_table ?t) (handempty ?r))
    :effect (and (not (toy_on_table ?t)) (not (handempty ?r)) (holdingToy ?r ?t))
  )

  (:action PlaceToyToBox
    :parameters (?r - robot ?t - toy ?b - box)
    :precondition (holdingToy ?r ?t)
    :effect (and (not (holdingToy ?r ?t)) (handempty ?r) (toy_in_box ?t ?b))
  )

  (:action PickWiperFromBox
    :parameters (?r - robot ?w - wiper ?b - box)
    :precondition (and (handempty ?r) (wiper_in_box ?w ?b) (box_at_center ?b))
    :effect (and (not (handempty ?r)) (not (wiper_in_box ?w ?b)) (holdingWiper ?r ?w))
  )

  (:action PickWiperFromTable
    :parameters (?r - robot ?w - wiper)
    :precondition (and (handempty ?r) (wiper_on_table ?w))
    :effect (and (not (handempty ?r)) (not (wiper_on_table ?w)) (holdingWiper ?r ?w))
  )

  (:action PlaceWiperAtTable
    :parameters (?r - robot ?w - wiper)
    :precondition (holdingWiper ?r ?w)
    :effect (and (not (holdingWiper ?r ?w)) (handempty ?r) (wiper_on_table ?w))
  )

  (:action PlaceWiperToBox
    :parameters (?r - robot ?w - wiper ?b - box)
    :precondition (holdingWiper ?r ?w)
    :effect (and (not (holdingWiper ?r ?w)) (handempty ?r) (wiper_in_box ?w ?b))
  )

  (:action PushBoxOut
    :parameters (?r - robot ?b - box)
    :precondition (and (box_at_center ?b) (handempty ?r))
    :effect (and (not (box_at_center ?b)) (box_at_side ?b))
  )

  (:action PullBoxIn
    :parameters (?r - robot ?b - box)
    :precondition (and (box_at_side ?b) (handempty ?r))
    :effect (and (not (box_at_side ?b)) (box_at_center ?b))
  )

  (:action WipeTable
    :parameters (?r - robot ?w - wiper ?b - box ?tb - table)
    :precondition (and (box_at_side ?b) (No_toy_at_table) (holdingWiper ?r ?w))
    :effect (table_clean ?tb)
  )

  (:action AchieveGoal
    :parameters (?r - robot ?tb - table)
    :precondition (and)
    :effect (goalAchieved)
  )
)